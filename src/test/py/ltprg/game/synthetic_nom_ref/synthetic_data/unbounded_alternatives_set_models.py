

# Trains model. Also assesses mean loss, accuracy, and KL-divergence from 
#       gold-standard S1 distribution (S1 produced by performing RSA using 
#       ground-truth lexicon) on full train and held-out validation sets
#       every n epochs.
#
# Model descriptions
# - EXPLICIT RSA MODEL ('ersa'): given an object embedding, neural network
#               produces truthiness vals between 0 and 1 for each 
#               utterance in the alternatives set. Each object in a trial 
#               is fed through the network, producing a lexicon that is 
#               then fed to RSA. RSA returns a level-1 speaker distribution, 
#               P(u | target, context, L)
# - NEURAL NETWORK WITH CONTEXT MODEL ('nnwc'): produces distribution over 
#               utterances in the fixed alternatives set given a trial's 
#               concatenated object embeddings, with target object in final 
#               position
# - NEURAL NETWORK WITHOUT CONTEXT MODEL ('nnwoc') produces distribution 
#               over utterances given target object emebdding only
#
# TODO: Add commandline args
#       Switch to inplace where possible
#       Add cuda support (though not really necessary yet given training speed)
#       Troubleshoot ReLU nan gradient issue
#       Add mini-batches (currently sz 1) (lower priority)
#       Add image embedding options (lower priority)