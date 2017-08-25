import torch.cuda as cuda
if cuda.is_available():
    print "CUDA is available!"
else:
    print "CUDA is unavailable :("
