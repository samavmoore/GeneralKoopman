import torch

def K_blocks(omegas, delta_t):
    '''
        Purpose: Make a dictionary containing all of the blocks of our koopman operator. The values from this dictionary are
        later passed into sci_py's block_diag function to construct K

        Inputs: Estimated real part (omegas[even]) and imaginary part (omegas[odd]). Must be an even number of elements in omegas
                    
        Outputs: Dictionary with keys block# and values 2x2 tensor with entries exp(real*dt)*[[cos(imag*dt), -sin(imag*dy)],[sin(imag*dt), cos(imag*dt)]]

    '''
    #initialize dictionary
    blocks = {}

    for i in range(0, len(omegas)-1, 2):

        # get real part
        R_part = omegas[i]
        # get imaginary part
        I_part = omegas[i+1]
    
        dt = delta_t

        growth_decay = torch.exp(R_part*dt)
        
        oscillations = torch.tensor([[torch.cos(I_part*dt), -torch.sin(I_part*dt)], \
                                    [torch.sin(I_part*dt), torch.cos(I_part*dt)]])

        oscillations = oscillations.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # store these blocks in the dictionary
        blocks[f"block {i}"] = torch.mul(oscillations, growth_decay)

    return blocks
