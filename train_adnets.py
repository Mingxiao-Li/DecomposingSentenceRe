from utils import to_device
from utils import build_dictionary,my_collate
from DataGenerator import DataGenerator
from torch.utils.data import DataLoader
from losses import *
from torch import optim
from ADNet import ADNet
from torch.autograd import Variable
from EarlyStopping import EarlyStopping
import torch
import numpy as np

def control_training_paramters(submodel,trainable):
    for para in submodel.parameters():
        para.requires_grad = trainable

def train(token2id, train_data,valid_data,lr_dis,lr_moti,lr_ge,batch_size,
          epochs, model,loss,weight_clipping=0.01,rho=0.02,patience = 10):

    dataset = DataGenerator(token2id,train_data)
    dataloader = DataLoader(dataset,batch_size=batch_size, collate_fn=my_collate)

    dataset_valid = DataGenerator(token2id,valid_data)
    dataloader_valid = DataLoader(dataset_valid,batch_size=batch_size,collate_fn=my_collate)

    early_stopping = EarlyStopping(patience=patience,verbose=True)

    model = to_device(model)

    if loss == "FisherGAN":
        Lambda = torch.FloatTensor([0])
        Lambda = to_device(Lambda)
        Lambda = Variable(Lambda,requires_grad = True)

        model_optimizer_dis_moti = optim.Adam([{"params": model.discriminator.parameters()},
                                               {"params": model.motivator.parameters(), "lr": lr_moti}],
                                                 lr=lr_dis)
        model_optimizer_ge = optim.Adam(model.parameters(), lr=lr_ge)

    else:
        model_optimizer_dis_moti = optim.RMSprop([{"params":model.discriminator.parameters()},
                                                  {"params":model.motivator.parameters(),"lr":lr_moti}],
                                                    lr = lr_dis)
        model_optimizer_ge = optim.RMSprop(model.parameters(),lr = lr_ge)

    re_loss = SequenceReconstructionLoss()

    if loss == "LeastSquare":
        dis_loss = LSGANDiscriminatorLoss()
        moti_loss = LSGANMotivatorLoss()
        dis_loss_adv = LSGANDiscriminatorLossAdv()
        moti_loss_adv = LSGANMotivatorLossAdv()

    elif loss == "WGAN":
        dis_loss = WGANDiscriminatorLoss()
        moti_loss = WGANMotivatorLoss()
        dis_loss_adv = WGANDiscriminatorLossAdv(0.5)
        moti_loss_adv = WGANMotivatorLossAdv(0.5)

    elif loss == "FisherGAN":
        dis_loss = FisherGANDiscriminatorLoss(Lambda,rho)
        moti_loss = FisherGANMotivatorLoss(Lambda,rho)
        dis_loss_adv = FisherGANDiscriminatorLossAdv()
        moti_loss_adv = FisherGANMotivatorLossAdv()

    for epoch in range(1,epochs+1):
        print("Epoch {}".format(epoch))
        data_2 = []

        for i, data in enumerate(dataloader):
            if (i == 0 or i%10 == 1) and i != 1:
                print("*"*80)
                print("START TRAINING DISCRIMINATOR AND MOTIVATOR")

            model_optimizer_dis_moti.zero_grad()

            data = to_device(data)
            x,x_len,y,t = data

            data_2.append(data)

            output = model(x,t,x_len)
            # train discriminator and motivator.

            dis_loss_value = dis_loss(output["dis_out"],y)
            moti_loss_value = moti_loss(output["moti_out"],y)
            loss_1 = dis_loss_value + moti_loss_value

            print("Epoch {}: loss of discriminator {},\n "
                  "        loss of motivator {}".
                  format(epoch,dis_loss_value.item(),moti_loss_value.item()))


            loss_1.backward()
            model_optimizer_dis_moti.step()

            if loss == "WGAN":

                for p in model.discriminator.parameters():
                    p.data.clamp_(-weight_clipping,weight_clipping)
                for p in model.motivator.parameters():
                    p.data.clamp_(-weight_clipping,weight_clipping)

            elif loss == "FisherGAN":
                Lambda.data += rho*Lambda.grad.data
                Lambda.grad.data.zero_()


            if i % 10 == 0 and i != 0:
                # train encoder and generator
                # freeze discriminator and motivator
                print("*"*80)
                print("START TRAINING ENCODER AND DECODER")
                control_training_paramters(model.discriminator,False)
                control_training_paramters(model.motivator,False)

                train_losses = []
                valid_losses = []
                avg_train_losses = []
                avg_valid_losses = []

                for data_g in data_2:
                    model_optimizer_ge.zero_grad()

                    x, x_len, y, t = data_g
                    output_2 = model(x,t,x_len)

                    re_loss_value = re_loss(output_2["logits"],x)
                    dis_loss_value2 = dis_loss_adv(output_2["dis_out"],y)
                    moti_loss_value2 = moti_loss_adv(output_2["moti_out"],y)

                    loss_2 = re_loss_value + dis_loss_value2 + moti_loss_value2

                    print("Epoch {}: loss of generator {},\n"
                          "         loss of discriminator {},\n"
                          "         loss of motivator {}".format(epoch,re_loss_value.item(),
                           dis_loss_value2.item(),moti_loss_value2.item()))


                    loss_2.backward()
                    model_optimizer_ge.step()
                    train_losses.append(re_loss_value.item())

                for j,data_valid in enumerate(dataloader_valid):
                    data_valid = to_device(data_valid)
                    x_valid,x_len_valid,y_valid,t_valid = data_valid

                    output_valid = model(x_valid,t_valid,x_len_valid)
                    loss_valid = re_loss(output_valid["logits"],x_valid)
                    valid_losses.append(loss_valid.item())

                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                epoch_len = len(str(epochs))
                print("+"*50)
                print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}]' +
                             f'train_loss: {train_loss:.5f}' +
                             f' valid_loss: {valid_loss:.5f}')
                print(print_msg)
                print("+" * 50)

                train_losses = []
                valid_losses = []

                early_stopping(valid_loss,model)
                if early_stopping.early_stop:
                    print("Early stopping")

                data_2 = []
                control_training_paramters(model.discriminator, True)
                control_training_paramters(model.motivator, True)

    torch.save(model.state_dict(),"./ADnets")
    print("Model has been saved successfully !!")

if __name__ == "__main__":
    training_data = "sentiment_data/training_data_shuffle.csv"
    valid_data = "sentiment_data/test_data_shuffle.csv"
    token2id = build_dictionary(training_data)
    adnet = ADNet(input_size=16,hidden_size=16,sentiment_size=8,max_len=35,
                  vocab_size=len(token2id),output_size=1)

    train(token2id,training_data,valid_data,lr_ge=0.001,lr_moti=0.001,lr_dis=0.001,
              batch_size=32,epochs=10,model=adnet,loss="FisherGAN")

