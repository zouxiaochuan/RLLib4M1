import torch
from tqdm import tqdm
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from iree import runtime as ireert
import iree.compiler as ireec
from torch_mlir.passmanager import PassManager
import numpy as np
import os
import torchvision.models as models
import time


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.fc0 = torch.nn.Linear(in_dim, hidden_size)
        self.tanh0 = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.tanh1 = torch.nn.Tanh()
        # self.output0 = torch.nn.Sigmoid()#(hidden_size, 1)
        self.output0 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc0(x)
        x = self.tanh0(x)
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.output0(x)
        return x

    pass


def compile_model(model, input_shape):
    module = torch.jit.script(model)
    mb = ModuleBuilder()
    class_annotator = ClassAnnotator()
    class_annotator.exportNone(module._c._type())
    class_annotator.exportPath(module._c._type(), ["forward"])
    class_annotator.annotateArgs(
        module._c._type(),
        ["forward"],
        [
            None,
            (input_shape, torch.float32, True),
        ],
    )
    mb.import_module(module._c, class_annotator)

    with mb.module.context:
        pm = PassManager.parse(
            "torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline"
        )
        # pm = PassManager.parse(
        #     'torchscript-module-to-torch-backend-pipeline')
        pm.run(mb.module)
    
    flatbuffer_blob = ireec.compile_str(
        str(mb.module), target_backends=['vulkan']
    )
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    config = ireert.Config('vulkan')
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]

    return ModuleCompiled


if __name__ == '__main__':
    model = MLP(1024, 512)
    # model = models.resnet50(pretrained=True)
    model.eval()

    # input_shape = [1, 3, 224, 224]
    input_shape = [-1, 1024]
    # x = np.random.rand(1, 3, 224, 224).astype('float32')
    x = np.random.rand(32, 1024).astype('float32')

    module_compiled = compile_model(model, input_shape)

    num_steps = 10000
    start = time.time()
    for i in tqdm(range(num_steps)):
        result = module_compiled(x)
        pass

    print(f'{num_steps} cost {time.time()-start} seconds')
    pass
