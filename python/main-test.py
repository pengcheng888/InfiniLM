import infinicore

modelpath = r"/home/ubuntu/Music/worksapce_nn/InfiniCore/test/infinicore/ops/model.pt"
import torch
model_param = torch.load(modelpath)


class InfiniNet(infinicore.nn.Module):
    def __init__(self):
        super(InfiniNet, self).__init__()
        self.fc1 = infinicore.nn.Linear(10, 6, bias=False)
        self.fc2 = infinicore.nn.Linear(6, 1, bias=False)

    def forward(self, x):
        x = x.view((1, 10))
        output = self.fc2.forward(self.fc1.forward(x))
        return output

    def test(self):
        model = InfiniNet()
        print(model)

        def  showmodeules(model):
            subs = model._modules
            print(subs)
            for key,value in subs.items():
                showmodeules(value)

        showmodeules(model)
        exit()
        # torch.save(model.state_dict(), "model.pt")
        # print("-----> before \n", model.state_dict())
        
        model.load_state_dict(model_param)
        # print("-----> after \n", model.state_dict())

        print('----------- caculate ------------>')

        device_str = "cuda"
        model.to(device=device_str)
        x = torch.ones((1, 10), dtype=torch.float32, device=device_str)

        out = model.forward(x)
        print(out)

        infini_x = infinicore.convert_torch_to_infini_tensor(x)

        out = model.forward(infini_x)
        print("==============>")
        print(out)

    


if __name__ == '__main__':
    InfiniNet().test()
