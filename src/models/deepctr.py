import torch
from deepctr_torch.layers.interaction import InteractingLayer
import math
from src.config import Config

torch.set_num_threads(Config['training_setting']['num_threads'])

class BNLDR(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, use_bn=True):
        super(BNLDR, self).__init__()
        self.use_bn = use_bn
        self.linear = torch.nn.Linear(in_features, out_features)
        if use_bn:
            self.bn = torch.nn.BatchNorm1d(in_features)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        if self.use_bn:
            return self.relu(self.dropout(self.linear(self.bn(inputs))))
        else:
            return self.relu(self.dropout(self.linear(inputs)))


class O3Model(torch.nn.Module):
    def __init__(self, mapping_dict, continuous_variables, catagory_variables,
                embedding_size=16, last_mlp_size=64, omlp_use_bn=True, infer_use_bn=True):
        super(O3Model, self).__init__()
        self.continuous_variables = continuous_variables
        self.catagory_variables = catagory_variables
        self.embedding_size = embedding_size
        self.last_mlp_size = last_mlp_size
        self.omlp_use_bn = omlp_use_bn
        self.infer_use_bn = infer_use_bn
        self.o1_size = self.embedding_size * len(self.catagory_variables) + len(self.continuous_variables)
        self.o2_size = self.embedding_size * (len(self.catagory_variables) + len(self.continuous_variables))
        self.om_sizes = [int(self.o1_size / 2**i) for i in range(math.ceil(math.log2(self.o1_size//self.last_mlp_size)) + 2)]
        self.extracted_size = self.o1_size + self.o2_size + self.om_sizes[-1]
        self.inference_sizes = [int(self.extracted_size / 2**i) for i in range(math.ceil(math.log2(self.extracted_size//self.last_mlp_size)) + 2)]

        # layers
        self.emb_layers = torch.nn.ModuleDict({x: torch.nn.Embedding(len(mapping_dict[x]), embedding_size, padding_idx=0) for x in catagory_variables})
        self.o2_layer = InteractingLayer(embedding_size=embedding_size, head_num=2)
        self.omlp_layers = torch.nn.ModuleList([
            BNLDR(self.om_sizes[i], self.om_sizes[i+1], dropout=0.5, use_bn=self.omlp_use_bn)
            for i in range(len(self.om_sizes)-1)
            ])
        self.inference_layers = torch.nn.ModuleList([
            BNLDR(self.inference_sizes[i], self.inference_sizes[i+1], dropout=0.5, use_bn=self.infer_use_bn)
            for i in range(len(self.inference_sizes)-1)
            ])
        self.inference_layers.append(torch.nn.Linear(self.inference_sizes[-1], 1))

        # function
        self.sigmoid = torch.nn.Sigmoid()

        pass

    def forward(self, inputs):
        # feature extracting
        emb_tensor_list = [self.emb_layers[x](inputs[x].type(torch.LongTensor)) for x in self.catagory_variables]
        o1_tensor = torch.concat(emb_tensor_list  + [inputs[x].unsqueeze(1) for x in self.continuous_variables], dim=1)
        o2_input = torch.stack(
            emb_tensor_list + \
            [inputs[x].repeat_interleave(self.embedding_size).reshape(-1, self.embedding_size) for x in self.continuous_variables],
            dim=1)

        o2_tensor = self.o2_layer(o2_input).flatten(1)
        oh_tensor = self.omlp_layers[0](o1_tensor)
        for l in self.omlp_layers[1:]:
            oh_tensor = l(oh_tensor)

        extracted_input = torch.concat([o1_tensor, o2_tensor, oh_tensor], dim=1)
        # inference
        infer_tensor = self.inference_layers[0](extracted_input)
        for l in self.inference_layers[1:-1]:
            infer_tensor = l(infer_tensor)

        # probability
        return self.sigmoid(self.inference_layers[-1](infer_tensor))



if __name__ == "__main__":
    pass




# embedding_size = 16
# last_mlp_size = 64
# omlp_use_bn = True
# infer_use_bn = True
# o1_size = embedding_size * len(catagory_variables) + len(continuous_variables)
# o2_size = embedding_size * (len(catagory_variables) + len(continuous_variables))
# om_sizes = [int(o1_size / 2**i) for i in range(math.ceil(math.log2(o1_size//last_mlp_size)) + 2)]
# extracted_size = o1_size + o2_size + om_sizes[-1]
# inference_sizes = [int(extracted_size / 2**i) for i in range(math.ceil(math.log2(extracted_size//last_mlp_size)) + 2)]


# # layers
# emb_layers = torch.nn.ModuleDict({x: torch.nn.Embedding(len(mapping_dict[x]), embedding_size, padding_idx=0) for x in catagory_variables})
# o2_layer = InteractingLayer(embedding_size=embedding_size, head_num=2)
# omlp_layers = torch.nn.ModuleList([
#     BNLDR(om_sizes[i], om_sizes[i+1], dropout=0.5, use_bn=omlp_use_bn)
#     for i in range(len(om_sizes)-1)
#     ])
# inference_layers = torch.nn.ModuleList([
#     BNLDR(inference_sizes[i], inference_sizes[i+1], dropout=0.5, use_bn=infer_use_bn)
#     for i in range(len(inference_sizes)-1)
#     ])
# inference_layers.append(torch.nn.Linear(inference_sizes[-1], 1))

# # function
# dropout = torch.nn.Dropout(0.5)
# sigmoid = torch.nn.Sigmoid()

# # inputs['site_id'].unsqueeze(1).shape
# # emb_tensor_list[0].shape


# # feature extracting
# inputs = next(iter(train_loader))['inputs']
# emb_tensor_list = [emb_layers[x](inputs[x].type(torch.LongTensor)) for x in catagory_variables]

# print(len(emb_tensor_list)) # number of categorical  feature
# print(emb_tensor_list[0].shape) # bat, dim

# o1_tensor = torch.concat(emb_tensor_list  + [inputs[x].unsqueeze(1) for x in continuous_variables], dim=1)
# print(o1_tensor.shape) # number of categorical feat * embedding dim

# #  # 假设是时间步T1
# #  T1 = torch.tensor([[1, 2, 3],
# #                  [4, 5, 6],
# #                  [7, 8, 9]])
# # print(T1.shape)                 
# #  # 假设是时间步T2
# #  T2 = torch.tensor([[10, 20, 30],
# #                  [40, 50, 60],
# #                  [70, 80, 90]])


# #  print(torch.stack((T1,T2),dim=0).shape)
# # print(torch.stack((T1,T2),dim=1).shape)
# #  print(torch.stack((T1,T2),dim=2).shape)


# o2_input = torch.stack(
#     emb_tensor_list + \
#     [inputs[x].repeat_interleave(embedding_size).reshape(-1, embedding_size) for x in continuous_variables],
#     dim=1)
# print(o2_input.shape) # batch, num of feat, dim



# o2_tensor = o2_layer(o2_input).flatten(1)
# print(o2_tensor.shape) # number of categorical feat * embedding dim
# # o2_layer(o2_input).shape

# oh_tensor = omlp_layers[0](o1_tensor)
# for l in omlp_layers[1:]:
#     oh_tensor = l(oh_tensor)
# print(oh_tensor.shape)


# extracted_input = torch.concat([o1_tensor, o2_tensor, oh_tensor], dim=1)
# print(extracted_input.shape)


# # inference
# infer_tensor = inference_layers[0](extracted_input)
# for l in inference_layers[1:-1]:
#     infer_tensor = l(infer_tensor)
# print(infer_tensor.shape)

# # probability
# sigmoid(inference_layers[-1](infer_tensor))





# torch = { version = "2.0.1", source="torch"}
# torchaudio = { version = "0.12.1", source="torch"}
# torchvision = { version = "0.13.1", source="torch"}

# [[tool.poetry.source]]
# name = "torch"
# url = "https://download.pytorch.org/whl/cu116"
# priority = "explicit"
