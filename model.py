import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.signal import convolve2d
import external.poissonblending as blending
import numpy as np
import pdb

from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class ContextLoss( nn.Module ):
    def __init__( self ):
        super( ContextLoss, self ).__init__()
        
    def forward( self, generated, corrupted, weight_mask ):
        c_loss = weight_mask * F.l1_loss(generated, corrupted)
        c_loss = c_loss.mean(dim=[0, 1, 2, 3])
        return c_loss

class PriorLoss( nn.Module ):
    def __init__( self, discriminator, lamb = 0.003 ):
        super( PriorLoss, self ).__init__()
        self.discriminator = discriminator
        self.lamb = lamb
        
    def forward( self, generated):
        discriminated = self.discriminator(generated)
        one = torch.ones_like(discriminated)
        # Discriminator Loss (Original)
        #p_loss = self.lamb*torch.log(one - discriminated + 1e-12)
        #p_loss = p_loss.mean(dim=[0, 1, 2, 3])

        # Loss - Alternative for Improvement
        p_loss = self.lamb*torch.log(discriminated + 1e-12)
        p_loss = p_loss.mean(dim=[0, 1, 2, 3])

        return p_loss

        
class ModelInpaint():
    def __init__( self, args ):
        self.batch_size = args.batch_size
        self.z_dim = 100
        self.n_size = args.n_size
        self.per_iter_step = args.per_iter_step

        self.generator = torch.load( args.generator )
        self.generator.eval()
        self.discriminator = torch.load( args.discriminator )
        self.discriminator.eval()

        
        
    def create_weight_mask( self, unweighted_masks ):
        kernel = np.ones( ( self.n_size, self.n_size ),
                          dtype=np.float32 )
        kernel = kernel / np.sum( kernel )
        weight_masks = np.zeros( unweighted_masks.shape, dtype=np.float32 )
        for i in range( weight_masks.shape[ 0 ] ):
            for j in range( weight_masks.shape[ 1 ] ):
                weight_masks[ i, j ] = convolve2d( unweighted_masks[ i, j ],
                                                   kernel,
                                                   mode='same',
                                                   boundary='symm' )
        weight_masks = unweighted_masks * ( 1.0 - weight_masks )
        return Tensor( weight_masks )

    def postprocess( self, corrupted, masks, generated ):
        corrupted = corrupted * 0.5 + 0.5
        generated = generated * 0.5 + 0.5
        corrupted = corrupted.permute( 0, 3, 2, 1 ).cpu().numpy()
        processed = generated.permute( 0, 3, 2, 1 ).cpu().detach().numpy()
        masks = np.transpose( masks, axes=( 0, 3, 2, 1 ) )
        for i in range( len( processed ) ):
            processed[ i ]  = blending.blend( corrupted[ i ],
                                             processed[ i ],
                                             1 - masks[ i ] )
        processed = torch.tensor( processed ).permute( 0, 3, 2, 1 )
        return ( processed * 2.0 - 1.0 ).cuda()

    def inpaint( self, corrupted, masks ):
        z = torch.tensor( np.float32( np.random.randn( self.batch_size,
                                                       self.z_dim ) ) )
        weight_mask = self.create_weight_mask( masks )
        if cuda:
            z = z.cuda()
            corrupted = corrupted.cuda()
            weight_mask = weight_mask.cuda()
        z_init = z.clone()
        print( 'Before optimizing: ' )
        print( z_init )
        
        # define loss functions
        loss_context = ContextLoss()
        loss_prior = PriorLoss(self.discriminator, lamb=0.003)
        
        # define optimizer 
        optimizer = optim.Adam([z.requires_grad_()], lr=0.000001) 
        
        # update z towards z_hat (Closest)
        for i in range(self.per_iter_step):
            #z should be [-1, 1] how? 
#             z = z - z.min()
#             z = (z / z.max()) * 2 - 1
                       
            
            generated = self.generator(z)
            l_c = loss_context(generated, corrupted, weight_mask)
            l_p = loss_prior(generated)
            loss = l_c + l_p
           
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
                 
            if (i+1) in [1, 10, 20, 100, 500, 1000, 1500]:
                print("-------------------------")
                print("Iteration %d has loss % f (l_c: %f + l_p: %f" % (i+1, loss.item(), l_c.item(), l_p.item()))
                print(z.min(), z.max())
                
        print( 'After optimizing: ' )
        print( z )
        generated = self.generator( z )
        save_image( generated+torch.from_numpy(masks).cuda(),
                'temp/Noblended.png',
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )
        save_image(torch.from_numpy(masks).cuda(),
                'temp/Mask.png',
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )


        return generated, self.postprocess( corrupted, masks, generated )
