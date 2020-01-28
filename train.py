import os
import random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import pandas as pd
from model import *
from dataset import *


# Save Model
def save_model(state_dict, name, epoch, model_dir):
    model_name = 'cdcgan_{}_{}.pth'.format(name, epoch)
    model_path = os.path.join(model_dir, model_name)
    torch.save(state_dict, model_path)

def check_filepath(path):
    if os.path.isfile(path):
        return path
    return False

# Load Model for resuming
def load_model(filepath, neural_net, optimizer):
    ckpt = torch.load(filepath)
    neural_net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return neural_net, optimizer

# randomly flip some labels
def noisy_labels(labels, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * labels.shape[0])
    # choose labels to flip
    flip_ix = torch.multinomial(torch.ones(labels.shape[0]), n_select)
    # invert the labels in place
    labels[flip_ix] = 1.0 - labels[flip_ix]
    return labels

# Generate images for triplet Net
def generate_images_triplet_training(generator, all_conds, save_path):
    """
        Generate image in the folder train_new
        Each identities will have at least 3, up to 5, generated images added
        This train_new folder will be used for tripletNet
        generator: trained generator
        all_conds: all the condition of all the classes
        path: location to folder
    """
    train_save_path = save_path + 'train_new'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
    for _, dirs, _ in os.walk(train_save_path, topdown=True):
        for dir in dirs:
            dst_path = train_save_path + '\\' + dir
            rand = random.randint(6, 7)
            rand_noise = torch.randn(rand, int(opt.z_dim), 1, 1, device=torch.device("cuda:0"))
            for i in range(rand):
                cam_id = random.randint(1, 6)
                shot = random.randint(1, 6)
                frame = random.randint(100, 1000)
                test_noise = rand_noise[i]
                test_noise = test_noise.unsqueeze(0)
                test_cond = all_conds[dir]
                fake_test = generator(test_noise, test_cond)
                vutils.save_image(fake_test.detach(), 
                                    '%s/%s_c%ss%s_00%s_fake.png' % (dst_path, dir, 
                                                                    str(cam_id), str(shot), str(frame)), 
                                                                    normalize=True)
    print('====== Done Generating Images ======')


def train(opt):
    """
    Common Params:
        z_dim: latent dimension
        gf_dim: generator fc dimension
        df_dim: discriminator fc dimension
        color_dim: color channels
        attr_range: range value of each attribute (0: No, 1: Yes, 2: Young, etc.)
                    e.g Binary value --> attr_range = 2
        attr_size: size to use for each attribute
        n_attr: 27 attributes
    """

    # -------------------------- Prepare Dataset --------------------------- #
    torch.manual_seed(random.randint(1,10000))
    cudnn.benchmark = True
    device = torch.device("cuda:0") # Set to Cuda()

    # Path to Dataset and Attribute File
    os.chdir("..\\..\\..")
    data_dir = '\\Market-1501\\pytorch\\'
    attr_frame = os.curdir + '\\Thesis Project\\market_attribute_full.csv'
    attr_test_frame = os.curdir + '\\Thesis Project\\market_attribute_test_full.csv'

    # ====== Training Data for CDCGAN ====== #
    transform_train_gan = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), 
                                                 (0.5, 0.5, 0.5))]) # normalize with mean, std = 0.5 
    dataset_gan = GanReIdFolder(root=os.curdir + data_dir + 'train',
                                transform=transform_train_gan,
                                attr_frame=attr_frame)
    
    if opt.all == 1:
        print('=' * 10 + 'Train all' + '=' * 10)
        attr_frame = os.curdir + '\\Thesis Project\\market_attribute_full.csv'
        dataset_gan = GanReIdFolder(root=os.curdir + data_dir + 'train_all',
                                transform=transform_train_gan,
                                attr_frame=attr_frame)

    dataloader_gan = DataLoader(dataset_gan,
                                batch_size=opt.batch_size,
                                shuffle=True, num_workers=2)

    # ======= Testing Data for CDCGAN ======= #
    transform_test_gan = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), 
                                                 (0.5, 0.5, 0.5))])
    dataset_test_gan = GanReIdFolder(root=os.curdir + data_dir + 'query',
                                transform=transform_test_gan,
                                attr_frame=attr_test_frame)
    dataloader_test_gan = DataLoader(dataset_test_gan,
                                batch_size=opt.batch_size,
                                shuffle=True, num_workers=2)

    # ====== Parameters for CDCGAN ====== #
    z_dim = int(opt.z_dim)
    gf_dim = int(opt.gf_dim)
    df_dim = int(opt.df_dim)
    color_dim = 3
    attr_range = 4 # must always be 4
    attr_size = int(opt.attr_size)
    n_attr = 27

    # Class to Index Dataset
    class_to_idx = dataset_gan.class_to_idx
    class_to_idx_test = dataset_test_gan.class_to_idx

    # CDCGAN initializing
    dcgan = DCGAN(attr_range=attr_range, n_attr=n_attr, 
                    attr_size=attr_size)
    generator = dcgan.generator
    discriminator = dcgan.discriminator
    optimizerG = dcgan.optimizerG
    optimizerD = dcgan.optimizerD

    criterion = nn.BCELoss() # Binary CrossEntropy Loss

    # Fixed noise with size (batch_size, 100, 1, 1)
    fixed_noise = torch.randn(opt.batch_size, z_dim, 1, 1, device=device)
    
    # Fixed test noise
    test_ids = ['0028', '0118', '0002', '0820']
    if opt.all == 1:
        test_ids.extend(['0018', '0238'])
    test_noise = torch.randn(len(test_ids), z_dim, 1, 1, device=device)

    # Correct cond. list
    r_conditions = None

    # === onehot attribute layers --> size (4, 4, 1, 1)===
    onehot = torch.zeros(attr_range, attr_range) 
    onehot = onehot.scatter_(1, torch.LongTensor(
                [i for i in range(attr_range)]
                ).view(attr_range, 1), 1).view(attr_range, attr_range, 1, 1)

    # Real/Fake label for training
    real_label = 1
    fake_label = 0

    def make_correct_cond(attr_lst):
        """
            attribute layer --> each has size(batch_size, 4, 1, 1)
        """
        return [onehot[attr_lst[:, col]].to(device) for col in range(len(attr_lst[0]))]
    
    def make_wrong_cond(correct_cond_lst):
        """
            Take in the correct cond. list 
            -> Make a wrong cond. list using randperm
        """
        result = []
        for cond in correct_cond_lst:
            idx = torch.randperm(cond.nelement())
            w_cond = cond.view(-1)[idx].view(cond.size())
            result.append(w_cond)
        return result

    def get_correct_cond_by_idx(correct_cond_lst, idx):
        """
            Return correct cond. of a certain image/class
        """
        return [cond[idx] for cond in correct_cond_lst]

    # === Make all the training conditions into onehot encoding
    conds = [idx for idx in class_to_idx.values()]
    attrs = pd.read_csv(attr_frame)
    attrs_lst = torch.LongTensor([attrs.iloc[cond, 1:].values for cond in conds])
    onehot_attrs = make_correct_cond(attrs_lst)
    all_conds = {}
    for id in range(len(conds)):
        for key, value in class_to_idx.items():
            if value == conds[id]:
                all_conds[key] = get_correct_cond_by_idx(onehot_attrs, id)

    # === Make all the testing conditions into onehot encoding
    conds_test = [idx for idx in class_to_idx_test.values()]
    attrs_test = pd.read_csv(attr_test_frame)
    attrs_lst_test = torch.LongTensor([attrs_test.iloc[cond, 1:].values for cond in conds_test])
    onehot_attrs_test = make_correct_cond(attrs_lst_test)
    all_conds_test = {}
    for id in range(len(conds_test)):
        for key, value in class_to_idx_test.items():
            if value == conds_test[id]:
                all_conds_test[key] = get_correct_cond_by_idx(onehot_attrs_test, id)

    print('-------------------------- Train CDCGAN ------------------------------')
    
    # -------------------------- Train CDCGAN ------------------------------ #

    # Load Pre-trained model if opt.resume is triggered
    path_gen = os.path.join(opt.model_dir, 
                            'cdcgan_generator_{}.pth'.format(opt.resume))
    path_disc = os.path.join(opt.model_dir,
                            'cdcgan_discriminator_{}.pth'.format(opt.resume))
    if opt.resume > 0 and check_filepath(path_gen) and\
                                check_filepath(path_disc):
        print('========== Resuming Training ==========')
        generator, optimizerG = load_model(path_gen, generator, optimizerG)
        discriminator, optimizerD = load_model(path_disc, discriminator, optimizerD)
        opt.epochs += opt.resume
    else: print('========== Start Training ==========')

    step_sizeG = 20
    step_sizeD = 60
    gammaG = 0.4
    gammaD = 1.2
    # Makes Generator less powerful after certain epochs
    schedulerG = lr_scheduler.MultiStepLR(optimizer=optimizerG, 
                                    milestones=[30, 60, 220], gamma=gammaG)
    # Makes Discriminator more powerful after certain epochs
    schedulerD = lr_scheduler.MultiStepLR(optimizer=optimizerD, 
                                    milestones=[55, 105, 210, 300], gamma=gammaD)

    for epoch in range(opt.resume, opt.epochs):
        for i, data in enumerate(dataloader_gan, 0):
        # ------ Train Discriminator ------ #

            # Get real image, labels, attrs
            real = data[0].cuda()
            # Image Labels stored w/ idx 0 -> 750
            y = data[2]
            # Correct conditions
            r_conditions = make_correct_cond(y)

            # Wrong conditions
            w_conditions = make_wrong_cond(r_conditions)

            # Get current batch_size (sometimes current batch < target batch)
            batch_size = real.size(0)

            # Fixed real/fake labels with noisy
            # real_labels = torch.ones(batch_size).cuda()
            real_labels = torch.ones(batch_size).cuda() * 0.9
            fake_labels = torch.zeros(batch_size).cuda()

            discriminator.zero_grad() # zero out gradient
            # Train with Real Images + Correct Conditions
            output = discriminator(real, r_conditions).view(-1) 
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with Fakes
            # Generate Fakes
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = generator(noise, r_conditions)

            # Train with Fake Images + Correct Conditions
            output = discriminator(fake.detach(), r_conditions).view(-1)
            errD_fake = criterion(output, fake_labels) / 2
            errD_fake.backward()
            # D(G(z))
            D_G_z1 = output.mean().item()

            # Train with Real Images + Wrong Conditions
            output = discriminator(real, w_conditions).view(-1)
            errD_fake2 = criterion(output, fake_labels) / 2
            errD_fake2.backward()

            # Optimize Discriminator
            errD = errD_real + errD_fake + errD_fake2
            optimizerD.step()

        # ------ Train Generator ------ # 
            generator.zero_grad()

            # The good fake data should be classified as real. 
            # correct conditions are used
            output = discriminator(fake, r_conditions).view(-1)
            # One-side smooth used only 
            # -> the generator use label = 1 instead of 0.9
            real_labels.fill_(real_label)
            errG = criterion(output, real_labels)
            errG.backward()
            # D(G(z))
            D_G_z2 = output.mean().item()

            # Optimize Generator
            optimizerG.step()

            print('[%d/%d][%d/%d]   Loss_D: %.4f   Loss_G: %.4f   D(x): %.4f    D(G(z)): %.4f / %.4f    lrG/lrD: %.5f / %.5f'
              % (epoch+1, opt.epochs, i+1, len(dataloader_gan),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, schedulerG.get_lr()[0], schedulerD.get_lr()[0]))
        
        # Decay Lr
        schedulerG.step()
        schedulerD.step()
                     
        # Save generated images for testing
        isExists=os.path.exists(os.path.join(opt.out_dir,str(epoch+1)+'_epoch'))
        if not isExists:
            os.makedirs(os.path.join(opt.out_dir,str(epoch+1)+'_epoch'))

        # Test CDCGAN with IDs '0028', '0118', '0002', '0820':
        for id in range(len(test_ids)):
            t_noise = test_noise[id]
            t_noise = t_noise.unsqueeze(0)
            test_cond = all_conds[test_ids[id]]
            fake_test = generator(t_noise, test_cond)
            vutils.save_image(fake_test.detach(), '%s/%d_epoch/faketest_%s_%d.png' % 
                                    (opt.out_dir, epoch+1, test_ids[id], epoch+1),normalize=True)
        # -----------------------------------        

    # Save GAN Model to re-train
    save_model({
        'epoch': opt.epochs,
        'state_dict': dcgan.generator.state_dict(),
        'optimizer': optimizerG.state_dict(),
        'train_conditions': all_conds,
        'test_conditions': all_conds_test,
    }, 'generator', int(opt.epochs), opt.model_dir)
    save_model({
        'epoch': opt.epochs,
        'state_dict': dcgan.discriminator.state_dict(),
        'optimizer': optimizerD.state_dict(),
    }, 'discriminator', int(opt.epochs), opt.model_dir)

    # Generate additional data or not
    if opt.generate_images == 1:
        print('-------------------------- Generating Images ------------------------------')
        path = os.curdir + data_dir
        generate_images_triplet_training(generator, all_conds, path)

    # Train triplet network or not
    if opt.triplet == 0:
        return generator, None

    print('-------------------------- Train TripletNet ------------------------------')
    
    # -------------------------- Train TripletNet -------------------------- #
    """
    Since this TripletNet is trained after the generator can generate decent fakes
    (usually after 500 epochs from opt.epochs)
    Common Params:
        epochs: default is 1500
        new_batch_size: here will be opt.batch_size / 2 --> default is 32
    """
    # ====== Training Data for TripletNet ====== #
    resume_epochs = opt.resume_triplet
    epochs = opt.triplet_epochs + opt.resume_triplet
    new_batch_size = int(opt.batch_size / 2)
    # data_dir = '\\Market-1501\\pytorch\\'

    transform_train_triplet = transforms.Compose([
                                transforms.Resize((256,128), interpolation=3),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                    (0.229, 0.224, 0.225)),
                                Cutout()])
                                # RandomErasing()])
    dataset_triplet = TripletReIdFolder(os.curdir + data_dir + 'train_new',
                                        transform_train_triplet)
    dataloader_triplet = DataLoader(dataset_triplet,
                                batch_size=new_batch_size,
                                shuffle=True, num_workers=2)
    triplet_classes = dataset_triplet.classes
    
    # ====== TripletNet Initializing ====== #
    tripletNet = EmbeddingResnet(len(triplet_classes)).cuda()
    optimizerTriplet = optim.SGD([
             {'params': tripletNet.parameters(), 'lr': opt.lr},
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    schedulerTriplet = lr_scheduler.MultiStepLR(optimizerTriplet, 
                                                milestones=[40,60], gamma=0.1)
    
    # ====== Resumning Training TripletNet ====== #
    path_tripletnet = os.path.join(opt.model_dir, 
                            'cdcgan_tripletnet_{}.pth'.format(opt.resume_triplet))
    if opt.resume_triplet > 0 and check_filepath(path_tripletnet):
        tripletNet, optimizerTriplet = load_model(path_tripletnet, tripletNet, optimizerTriplet)
        print('========== Resuming Training TripletNet ==========')
    else:
        print('========== Start Training TripletNet ==========')
    
    for epoch in range(opt.resume_triplet, epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        schedulerTriplet.step()
        tripletNet.train()

        running_loss = 0.0
        running_corrects = 0.0

        for data in dataloader_triplet:
            inputs, labels, pos_imgs, pos_labels = data
            now_batch_size, channel, height, width = inputs.shape
            
            if now_batch_size < new_batch_size:
                continue

            # Processing Positive images and labels
            pos_imgs = pos_imgs.view(2 * new_batch_size, channel, height, width)
            pos_labels = pos_labels.repeat(2).reshape(2, new_batch_size)
            pos_labels = pos_labels.transpose(0,1).reshape(2 * new_batch_size)

            inputs = Variable(inputs.cuda())
            pos_imgs = Variable(pos_imgs.cuda())
            labels = Variable(labels.cuda())                    

            optimizerTriplet.zero_grad()

            # Processing Outputs
            outputs = tripletNet(inputs)
            pos_outputs = tripletNet(pos_imgs)
            neg_labels = pos_labels

            # ------------ Hard Negative Mining ------------
            neg_outputs = pos_outputs 
            rand = np.random.permutation(2 * new_batch_size)[0:opt.poolsize]
            neg_outputs = neg_outputs[rand,:] 
            neg_labels = neg_labels[rand]
            neg_outputs_t = neg_outputs.transpose(0,1)
            score = torch.mm(outputs.data, neg_outputs_t) # Cosine similarity
            # Sort score, higher score means -> different -> hard negatives
            score, rank = score.sort(dim=1, descending=True)
            labels_cpu = labels.cpu()
            neg_outputs_hard = torch.zeros(outputs.shape).cuda()
            for k in range(now_batch_size):
                hard = rank[k,:]
                for kk in hard:
                    now_label = neg_labels[kk] 
                    anchor_label = labels_cpu[k]
                    if now_label != anchor_label:
                        neg_outputs_hard[k,:] = neg_outputs[kk,:]
                        break

            # ------------ Hard Positive Mining ------------
            pos_outputs_hard = torch.zeros(outputs.shape).cuda()
            for j in range(now_batch_size):
                pos_data = pos_outputs[2*j:2*j+2, :]
                pos_data_t = pos_data.transpose(0,1)
                ff = outputs.data[j,:].reshape(1,-1)
                score = torch.mm(ff, pos_data_t)
                # Sort score, lower score means -> similar -> hard positives
                score, rank = score.sort(dim=1, descending=False)
                pos_outputs_hard[j,:] = pos_data[rank[0][0],:]

            # ------------ Calculate LOSS ------------
            reg = torch.sum((neg_score + 1)**2) + torch.sum((pos_score - 1)**2)
            pos_score = torch.sum( outputs * pos_outputs_hard, dim=1) 
            neg_score = torch.sum( outputs * neg_outputs_hard, dim=1)

            loss = torch.sum(F.relu(neg_score + opt.margin - pos_score)) 
            loss_triplet = loss + opt.alpha * reg

            loss_triplet.backward()
            optimizerTriplet.step()

            # ------------ Calculate Statistics ------------
            running_loss += loss_triplet.item()
            running_corrects += float(torch.sum(pos_score > neg_score + opt.margin))

        datasize = len(dataset_triplet)//new_batch_size * new_batch_size
        epoch_loss = running_loss / datasize
        epoch_acc = running_corrects / datasize
        
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        if epoch % 10 == 0:
            save_model({
                'epoch': epoch,
                'state_dict': tripletNet.state_dict(),
                'optimizer': optimizerTriplet.state_dict(),
            }, 'tripletnet', int(epoch), opt.model_dir)

    save_model({
        'epoch': epochs,
        'state_dict': tripletNet.state_dict(),
        'optimizer': optimizerTriplet.state_dict(),
    }, 'tripletnet', int(epochs), opt.model_dir)
    
    return generator, tripletNet
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--z_dim', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--gf_dim', type=int, default=64, help='size of first conv of generator')
    parser.add_argument('--df_dim', type=int, default=64, help='size of first conv of discriminator')
    parser.add_argument('--epochs', type=int, default=4000, help='number of epochs to train for')
    parser.add_argument('--out_dir', default=os.curdir + '\\Thesis Project\\ganreid_result', help='folder to output images')
    parser.add_argument('--model_dir', default=os.curdir + '\\Thesis Project\\model_dir', help='folder to save model')
    parser.add_argument('--resume', type=int, default=0, help='resume training and matching epochs')
    parser.add_argument('--test', type=bool, default=False, help='trigger test')
    parser.add_argument('--all', type=int, default=0, help='train all?')
    parser.add_argument('--margin', default=0.3, type=float, help='margin')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--name',default='ft_resnet50', type=str, help='output model name')
    parser.add_argument('--poolsize', default=64, type=int, help='poolsize')
    parser.add_argument('--alpha', default=0.0, type=float, help='regularization, push to -1')
    parser.add_argument('--generate_images', default=0, type=int, help='generate data for tripletNet')
    parser.add_argument('--attr_size', type=int, default=4, help='attribute size')
    # Triplet
    parser.add_argument('--triplet', default=1, type=int, help='train triplet or not?')
    parser.add_argument('--resume_triplet', type=int, default=100, help='resume training and matching epochs for TripletNet')
    parser.add_argument('--triplet_epochs', type=int, default=1500, help='Epochs for training triplet')
    
    opt = parser.parse_args()
    print(opt)

    train(opt)
