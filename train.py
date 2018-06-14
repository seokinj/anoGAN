import os
import scipy.misc
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch import optim
from torch.autograd import Variable
from utils import *
from model import *

def anomaly_score(G, D, ano_z, test_input):
	ano_G = G(ano_z)
	residual_loss = torch.sum(torch.abs(test_input-ano_G))
			
	feature_ano_G, _ = D(ano_G)
	feature_input, _ = D(test_input)
	disc_loss = torch.sum(torch.abs(feature_ano_G-feature_input))

	total_loss = (1.0-config.ano_para)*residual_loss + (config.ano_para)*disc_loss			
	return ano_G, total_loss	
		

if __name__ == "__main__":
	use_cuda = torch.cuda.is_available()
	gpu = 0

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, dest='config', help='the name of yaml file to set parameter', default='config.yaml')
	parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model", action='store_true', default=False)
	parser.add_argument('--anomaly', dest='anomaly', help="switch for anomaly detecting", action='store_true', default=True)
	parser.add_argument('--root_dir', type=str, dest='root_dir', help='the path of current directory')
	parser.add_argument('--train_dir', type=str, dest='train_dir', help='the path of train data')
	parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='the path of chekcpoint dir', default='checkpoint')
	parser.add_argument('--save_dir', type=str, dest='save_dir', help='the path of generated data dir', default='sample')
	parser.add_argument('--test_dir', type=str, dest='test_dir', help='the path of anomaly test data')
	parser.add_argument('--test_result_dir', type=str, dest='test_result_dir', help='the path of anomaly test result dir')

	args = parser.parse_args()
	config = Config(args.config)

	if not os.path.exists(args.save_dir):
		os.mkdir(os.path.join(args.root_dir, args.save_dir))
	transform = transforms.Compose([
        transforms.Scale(config.image_size),
        transforms.ToTensor(),                     
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	if config.dataset == 'mnist':
		train_loader = load_data(os.path.join(args.root_dir, args.train_dir), transform, 'mnist', config)
	elif config.dataset == 'celebA':
		train_loader = load_data(os.path.join(args.root_dir, args.train_dir), transform, 'celebA', config)
	elif config.dataset == 'cifar10':
		train_loader = load_data(os.path.join(args.root_dir, args.train_dir), transform, 'cifar10', config)

	G = Generator(config.z_dim, config.c_dim, config.gf_dim)
	D = Discriminator(config.c_dim, config.df_dim)

	if not args.pretrained:
		if use_cuda:
			G = G.cuda(gpu)
			D = D.cuda(gpu)

		# WHY BECLoss() - only need to determine fake/real for Discriminator
		criterion = nn.BCELoss()
		if use_cuda:
			criterion = criterion.cuda(gpu)

		optimizerD = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
		optimizerG = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
	
		batch_time = AverageMeter()
		data_time = AverageMeter()
		D_losses = AverageMeter()
		G_losses = AverageMeter()

		fixed_noise = torch.FloatTensor(8 * 8, config.z_dim, 1, 1).normal_(0, 1)
		if use_cuda:
			fixed_noise = fixed_noise.cuda(gpu)
		with torch.no_grad():
			fixed_noisev = fixed_noise

		end = time.time()
		
		D.train()
		G.train()
		D_loss_list = []
		G_loss_list = []
	
		for epoch in range(config.epoches):
			for i, (input, label, _) in enumerate(train_loader):
				# Update 'D' : max log(D(x)) + log(1-D(G(z)))
				data_time.update(time.time()-end)
			
				batch_size = input.size(0)
				label_real = torch.ones(batch_size)
				label_fake = torch.zeros(batch_size)
				if use_cuda:
					input = input.cuda(gpu)
					label_real = label_real.cuda(gpu)
					label_fake = label_fake.cuda(gpu)
			
				_, D_real = D(input)
				D_real = criterion(D_real, label_real)
		
				noise = torch.randn((batch_size, config.z_dim)).view(-1, config.z_dim, 1, 1)
				if use_cuda:
					noise = noise.cuda()
				G_sample = G(noise)
	
				_, D_fake = D(G_sample)	
				D_fake = criterion(D_fake, label_fake)
		
				D_loss = D_real + D_fake
				D_losses.update(D_loss.data[0])
				D.zero_grad()
				D_loss.backward()
				optimizerD.step()

				# Update 'G' : max log(D(G(z)))
				noise = torch.randn((batch_size, config.z_dim)).view(-1, config.z_dim, 1, 1)
				if use_cuda:
					noise = noise.cuda()
				G_sample = G(noise)
		
				_, D_fake = D(G_sample)	
				G_loss = criterion(D_fake, label_real)
				G_losses.update(G_loss.data[0])
			
				D.zero_grad()
				G.zero_grad()
				G_loss.backward()
				optimizerG.step()

				batch_time.update(time.time()-end)
				end = time.time()
		
				# log every 100th train data of train_loader - display(100)	
				if (i+1) % config.display == 0:
					print_log(epoch+1, config.epoches, i+1, len(train_loader), config.base_lr, config.display, batch_time, data_time, D_losses, G_losses)
					# Is it Continous ???
					batch_time.reset()
					data_time.reset()
				# log every 1 epoch (all of train_loader)
				elif (i+1) == len(train_loader):
					print_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
	                          (i + 1) % config.display, batch_time, data_time, D_losses, G_losses)
					batch_time.reset()
					data_time.reset()

			# log every 1 epoch
			D_loss_list.append(D_losses.avg)
			G_loss_list.append(G_losses.avg)
			D_losses.reset()
			G_losses.reset()

			plot_result(G, fixed_noisev, config.image_size, epoch + 1, args.save_dir, 'dcgan', is_gray=(config.c_dim == 1))
			plot_loss(epoch+1, config.epoches, args.save_dir, d_loss=D_loss_list, g_loss=G_loss_list)
			# save the D and G.
			save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(),}, os.path.join(args.checkpoint_dir, 'D_epoch_{}'.format(epoch)))
			save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(),}, os.path.join(args.checkpoint_dir, 'G_epoch_{}'.format(epoch)))
	
		create_gif(config.epoches, args.save_dir, 'dcgan')

	## Pretrained Model"
	else:
		print("Use Pretrained Model")
		if use_cuda:
			G = G.cuda()
			D = D.cuda()
		G.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "G_epoch_"+ str(config.epoches-1) + ".pth.tar"))['state_dict'])
		D.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "D_epoch_"+ str(config.epoches-1) + ".pth.tar"))['state_dict'])

	## TEST ANOMALY ##
	if args.anomaly:
		if not os.path.exists(args.test_dir):
			os.mkdir(os.path.join(args.root_dir, args.test_dir))
		if not os.path.exists(args.test_result_dir):
			os.mkdir(os.path.join(args.root_dir, args.test_result_dir))
		test_loader = load_data(args.test_dir, transform, 'test', config)

		save_epoch = [0, int(config.test_epoches/2), config.test_epoches-1]
		for i, (test_input, label, path) in enumerate(test_loader):
			print("{} st test sample : {}".format(i+1, path))
			ano_z = torch.randn((1, config.z_dim)).view(-1, config.z_dim, 1, 1)
			if use_cuda:
				ano_z = ano_z.cuda(gpu)
				ano_z = ano_z.requires_grad_()	
			optimizerZ = torch.optim.Adam([ano_z], lr=config.base_lr, betas=(config.beta1, 0.999)) # [ano_z] : to make list for parameter
			z_loss_list = []
			for epoch in range(config.test_epoches):
				if use_cuda:
					test_input = test_input.cuda()
				ano_G, loss = anomaly_score(G, D, ano_z, test_input)
				z_loss_list.append(loss)
				optimizerZ.zero_grad()
				loss.backward()
				optimizerZ.step()
				if epoch in save_epoch:
					"""
					errors = ano_G - test_input
					print("ano_G : {}".format(ano_G.shape))
					print("test_input : {}".format(test_input.shape))
					samples = np.squeeze(ano_G.detach())
					errors = np.squeeze(errors.detach())
					samples = (np.array(samples)+1)*127.5 # image * 127.5 + 127.5 = (image+1)*127.5 
					print(samples.shape)
					errors = (np.array(errors)+1)*127.5
			
					filename = ['AD_'+str(i)+'_'+str(epoch)+'.jpg', 'AD_error_'+str(i)+'_'+str(epoch)+'.jpg']
					scipy.misc.imsave(os.path.join(args.test_result_dir, filename[0]),samples)
					scipy.misc.imsave(os.path.join(args.test_result_dir,filename[1]),errors)
					"""
					print("{} epoch anomaly score of {}: {}".format(epoch+1, path[0].split('/')[-1], z_loss_list[-1]))
			# need filename!
			# plot_result(G, ano_z, config.image_size, epoch+1, args.test_result_dir, 'anomaly', is_gray=(config.c_dim == 1)) # plot samples
			# plot_loss(epoch+1, config.epoches, args.test_result_dir, z_loss=z_loss_list) # plot z loss
		
		#create_gif(config.test_epoches, args.test_result_dir, 'anomaly')
