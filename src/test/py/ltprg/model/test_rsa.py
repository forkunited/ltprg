import unittest
import torch
import torch.nn as nn
import ltprg.model.rsa as rsa
from torch.autograd import Variable
from ltprg.model.dist import Categorical


class TestRSA(unittest.TestCase):

    # Test against model from
    # http://gscontras.github.io/ESSLLI-2016/chapters/1-introduction.html
    def test_rsa(self):
        class World:
            SQUARE_BLUE = 0
            CIRCLE_BLUE = 1
            SQUARE_GREEN = 2

            @staticmethod
            def get_all():
                return [World.SQUARE_BLUE, World.CIRCLE_BLUE, World.SQUARE_GREEN]

            @staticmethod
            def get_index(world):
                return world.data

            @staticmethod
            def get_str(world):
                if world == World.SQUARE_BLUE:
                    return "SQUARE-BLUE"
                elif world == World.CIRCLE_BLUE:
                    return "CIRCLE-BLUE"
                elif world == World.SQUARE_GREEN:
                    return "SQUARE-GREEN"

        class Utterance:
            BLUE = 0
            GREEN = 1
            SQUARE = 2
            CIRCLE = 3

            @staticmethod
            def get_all():
                return [Utterance.BLUE, Utterance.GREEN, Utterance.SQUARE, Utterance.CIRCLE]

            @staticmethod
            def get_index(utterance):
                return utterance.data

            @staticmethod
            def get_str(utterance):
                if utterance == Utterance.BLUE:
                    return "blue"
                elif utterance == Utterance.GREEN:
                    return "green"
                elif utterance == Utterance.SQUARE:
                    return "square"
                elif utterance == Utterance.CIRCLE:
                    return "circle"


        class WorldPriorFn(nn.Module):
            def __init__(self):
                super(WorldPriorFn, self).__init__()

            def forward(self, observation):
                """
                Returns:
                    A vectorized batch of world distributions
                """
                vs = torch.LongTensor(World.get_all()).unsqueeze(0).repeat(observation.size(0),1)
                return Categorical(Variable(vs))

            def get_index(self, world, observation, support):
                return World.get_index(world), False, None

        class UtterancePriorFn(nn.Module):
            def __init__(self):
              super(UtterancePriorFn, self).__init__()

            def forward(self, observation):
                """
                Returns:
                    A vectorized batch of utterance distributions
                """
                vs = torch.LongTensor(Utterance.get_all()).unsqueeze(0).repeat(observation.size(0),1)
                return Categorical(Variable(vs))

            def get_index(self, utterance, observation, support):
                return Utterance.get_index(utterance), False, None

        def meaning_fn(utterance, world, observation):
            """
            Computes whether utterances apply to worlds
            Args:
                utterance (:obj:`utterance batch_like`):
                    (Batch size) x (Utterance prior size) x (Utterance) array
                        of utterances
                world (:obj:`world batch_like`):
                    (Batch size) x (World prior size) x (World) array
                        of worlds

                observation (:obj:`observation batch_like`)
            Returns:
                (Batch size) x (Utterance prior size) x (World prior size)
                    tensor of meaning values
            """
            batch_size = utterance.size(0)
            meaning = torch.zeros(batch_size, utterance.size(1), world.size(1))
            for b in range(batch_size):
                for u_i in range(meaning.size(1)):
                    for w_i in range(meaning.size(2)):
                        u = utterance[b,u_i].data[0]
                        w = world[b,w_i].data[0]
                        if (u == Utterance.BLUE and (w == World.CIRCLE_BLUE or w == World.SQUARE_BLUE)) \
                            or (u == Utterance.GREEN and (w == World.SQUARE_GREEN)) \
                            or (u == Utterance.CIRCLE and (w == World.CIRCLE_BLUE)) \
                            or (u == Utterance.SQUARE and (w == World.SQUARE_BLUE or w == World.SQUARE_GREEN)):
                            meaning[b, u_i, w_i] = 1.0
            return Variable(meaning)

        world_prior_fn = WorldPriorFn()
        utterance_prior_fn = UtterancePriorFn()

        L0 = rsa.L("L_0", 0, meaning_fn, world_prior_fn, utterance_prior_fn)
        S1 = rsa.S("S_1", 1, meaning_fn, world_prior_fn, utterance_prior_fn)
        L1 = rsa.L("L_1", 1, meaning_fn, world_prior_fn, utterance_prior_fn)

        utterance_batch = Variable(torch.LongTensor([Utterance.BLUE, Utterance.GREEN, Utterance.SQUARE, Utterance.CIRCLE]))

        l0_dist = L0(utterance_batch)
        l0_dist_support = l0_dist.support()
        l0_dist_p = l0_dist.p()

        print "L0 Distributions"
        print "----------------"
        for b in range(utterance_batch.size(0)):
            s = Utterance.get_str(utterance_batch[b].data[0]) + "\t::- "
            for i in range(l0_dist_p.size(1)):
                s += World.get_str(l0_dist_support[b,i].data[0]) + ": " + str(l0_dist_p[b,i].data[0]) + "\t"
            print s
        print "\n"

        l1_dist = L1(utterance_batch)
        l1_dist_support = l1_dist.support()
        l1_dist_p = l1_dist.p()

        print "L1 Distributions"
        print "----------------"
        for b in range(utterance_batch.size(0)):
            s = Utterance.get_str(utterance_batch[b].data[0]) + "\t::- "
            for i in range(l1_dist_p.size(1)):
                s += World.get_str(l1_dist_support[b,i].data[0]) + ": " + str(l1_dist_p[b,i].data[0]) + "\t"
            print s
        print "\n"

        world_batch = Variable(torch.LongTensor([World.SQUARE_BLUE, World.CIRCLE_BLUE, World.SQUARE_GREEN]))

        s1_dist = S1(world_batch)
        s1_dist_support = s1_dist.support()
        s1_dist_p = s1_dist.p()

        print "S1 Distributions"
        print "----------------"
        for b in range(world_batch.size(0)):
            s = World.get_str(world_batch[b].data[0]) + "\t::- "
            for i in range(s1_dist_p.size(1)):
                s += Utterance.get_str(s1_dist_support[b,i].data[0]) + ": " + str(s1_dist_p[b,i].data[0]) + "\t"
            print s
        print "\n"


if __name__ == '__main__':
    unittest.main()
