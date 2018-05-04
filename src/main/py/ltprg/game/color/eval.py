import torch
import colorsys
import numpy as np
import math
import os
import os.path
from torch.autograd import Variable
from mung.torch_ext.eval import Evaluation
from ltprg.model.seq import SamplingMode
from ltprg.util.img import make_rgb_img, make_gray_img
from ltprg.game.color.util import construct_color_space
from skimage.color import rgb2lab, lab2rgb
from PIL import Image, ImageDraw, ImageFont

COLORS_PER_DIM=50
CAPTION_IMG_WIDTH=140
CAPTION_IMG_HEIGHT=70
COLOR_IMG_WIDTH=140
COLOR_IMG_HEIGHT=140

class ColorMeaningPlot(Evaluation):
    def __init__(self, name, data, data_parameters, output_file_prefix, utts_file_path, s_model, utt_sampling_mode, \
                 meaning_fn, utt_data, utts_count=15, max_utt_length=7):
        super(ColorMeaningPlot, self).__init__(name, data, data_parameters)

        self._output_file_prefix = output_file_prefix
        self._utts_file_path = utts_file_path
        self._meaning_fn = meaning_fn
        self._utt_data = utt_data
        self._iteration = 0
        self._sample_seqs_to_file(utts_file_path, s_model, utt_sampling_mode, utts_count, max_utt_length)

    def _run_batch(self, model, batch):
        pass

    def _aggregate_batch(self, agg, batch_result):
        pass

    def _initialize_result(self):
        pass

    def _finalize_result(self, result):
        output_path = self._output_file_prefix + "_" + str(self._iteration) + ".png"
        self._plot_meaning(self._utts_file_path, output_path, self._meaning_fn, self._utt_data)
        self._iteration += 1
        return output_path

    def _make_text_img(self, text, width=140, height=140):
        img = make_gray_img(np.ones(shape=(width,height)), width=width, height=height)
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype("arial.ttf", 16)
        draw.text((0, height/2), text, (0,0,0))#, font=font)
        return img

    def _load_utts_from_file(self, file_path):
        samples, lens = torch.load(file_path)
        return samples, lens

    def _make_utt_str(self, seq, length, utt_data):
        return " ".join([utt_data.get_feature_token(seq[k]).get_value() for k in range(length)])

    def _save_plot(self, output_path, caption_imgs, color_imgs, caption_width=140, caption_height=140, color_width=140, color_height=140):
        caption_height = 0 # FIXME This is here to remove captions for now
        num_rows = int(math.floor(len(caption_imgs)**0.5))
        num_columns = int(math.ceil(len(caption_imgs)/num_rows))
        full_height = num_rows*(caption_height+color_height)
        full_width = num_columns*color_width

        full_img = Image.new('RGB', (full_width, full_height))
        for i in range(len(caption_imgs)):
            x_pos = (i % num_columns)*max(color_width, caption_width)
            y_pos = (i / num_columns)*(color_height+caption_height)
            #full_img.paste(caption_imgs[i], (x_pos, y_pos))
            full_img.paste(color_imgs[i], (x_pos, y_pos+caption_height))
        full_img.save(output_path)

    def _sample_seqs_to_file(self, file_path, model, sampling_mode, n, max_length):
        samples = None
        if sampling_mode == SamplingMode.FORWARD:
            samples = model.sample(n_per_input=n, max_length=max_length)
        else:
            samples = model.beam_search(beam_size=n, max_length=max_length)

        # Sample : Seq length x batch size
        # Lens : Batch size array
        sample, lens, scores = samples[0]
        sample = sample.transpose(0,1)
        utt_str = ""
        for u in range(sample.size(0)):
            utt_str += "(" + str(u) + ") " + self._make_utt_str(sample[u], lens[u], self._utt_data) + "\n"

        torch.save((sample, lens), file_path)
        with open(file_path + "_str.txt", "w") as text_file:
            text_file.write(utt_str)


    def _plot_meaning(self, utterance_path, output_path, meaning_fn, utt_data):
        utterances = self._load_utts_from_file(utterance_path)
        colors = construct_color_space(n_per_dim=COLORS_PER_DIM)
        world_idx = torch.arange(0, colors.size(0)).long()

        if meaning_fn.on_gpu():
            world_idx = world_idx.cuda()
            colors = colors.cuda()

        meanings = meaning_fn((Variable(utterances[0].unsqueeze(0)), utterances[1].unsqueeze(0)), Variable(world_idx.unsqueeze(0)), Variable(colors.view(1,colors.size(0)*colors.size(1)))).squeeze() # utterances x colors
        meanings = meanings.data

        colors_rgb = construct_color_space(n_per_dim=COLORS_PER_DIM, rgb=True)
        caption_imgs = [self._make_text_img("World", width=CAPTION_IMG_WIDTH, height=CAPTION_IMG_HEIGHT)]
        color_imgs = [make_rgb_img(colors_rgb.view(COLORS_PER_DIM, COLORS_PER_DIM, 3).numpy(), width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT)]
        for u in range(meanings.size(0)): # For each utterance
            utt_str = self._make_utt_str(utterances[0][u], utterances[1][u], utt_data)
            caption_imgs.append(self._make_text_img(utt_str, width=CAPTION_IMG_WIDTH, height=CAPTION_IMG_HEIGHT))
            color_imgs.append(make_gray_img(meanings[u].contiguous().view(COLORS_PER_DIM,COLORS_PER_DIM).cpu().numpy(), width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT))
        self._save_plot(output_path, caption_imgs, color_imgs, caption_width=CAPTION_IMG_WIDTH, caption_height=CAPTION_IMG_HEIGHT, color_width=COLOR_IMG_WIDTH, color_height=COLOR_IMG_HEIGHT)



