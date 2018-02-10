import sys
import numpy as np
import json
import torch
from os.path import join
from torch.autograd import Variable
from mung.data import DataSet
from mung.feature import MultiviewDataSet
from ltprg.model.meaning import MeaningModel
from ltprg.model.seq import SequenceModel
from ltprg.game.color.properties.colorspace_conversions import hsls_to_rgbs, rgbs_to_labs

gpu = True
seed = 1
data_color_dir = sys.argv[1]
data_utterance_dir = sys.argv[2]
model_color_s0_path = sys.argv[3]
model_color_meaning_path = sys.argv[4]
trials_dir = sys.argv[5]
output_dir = sys.argv[6]
grid_dim = int(sys.argv[7])


if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# NOTE: Unncecessary to load all this.  Just need the feature set for the
# utterance token vocabulary.  But doing it this way for now because it's easy
D_COLOR = MultiviewDataSet.load(data_color_dir, dfmatseq_paths={ "utterance" : data_utterance_dir })
MODEL_COLOR_S0 = SequenceModel.load(model_color_s0_path)
MODEL_COLOR_MEANING = MeaningModel.load(model_color_meaning_path)
UTTERANCE_PRIOR_SAMPLES_PER_COLOR = 3
MAX_UTTERANCE_LENGTH = 24
SPEAKER_ALPHA = 16.0

OBJECT_WIDTH = 3
OBJECT_HEIGHT = 3

ALL_POSITIONS = set([])
for i in range(grid_dim):
    for j in range(grid_dim):
        ALL_POSITIONS.add((i,j))

ARG_COLOR = "[color]"
ARG_POSITION = "[pos]"
ARG_POSITION_CORNER = "[pos_c]"
ARG_POSITION_ROW = "[pos_r]"

COLUMN_LEFT_DESCRIPTIONS = ["left"]
COLUMN_RIGHT_DESCRIPTIONS = ["right"]
COLUMN_MIDDLE_DESCRIPTIONS = ["center", "middle"]

ROW_TOP_DESCRIPTIONS = ["top", "upper"]
ROW_BOTTOM_DESCRIPTIONS = ["bottom", "lower"]
ROW_MIDDLE_DESCRIPTIONS = ["center", "middle"]

class UtteranceTemplate:
    def __init__(self, form):
        self._form = form

    def apply(self, args):
        utt_str = self._form
        for arg in args:
            utt_str = utt_str.replace(arg, str(args[arg]))
        return utt_str

    def has_corner_arg(self):
        return (ARG_POSITION_CORNER in self._form)

    def sample_args(self, shape):
        template_args = dict()
        if ARG_COLOR in self._form:
            template_args[ARG_COLOR] = shape.get_color().sample_description()
        if ARG_POSITION in self._form:
            template_args[ARG_POSITION] = shape.sample_position_description()
        if ARG_POSITION_ROW in self._form:
            template_args[ARG_POSITION_ROW] = shape.sample_row_description()
        if ARG_POSITION_CORNER in self._form:
            template_args[ARG_POSITION_CORNER] = shape.sample_position_description()
        return template_args

class Utterance:
    def __init__(self, template, template_args):
        self._template = template
        self._template_args = template_args

    def __str__(self):
        return self._template.apply(self._template_args)

    def __eq__(self, o):
        return str(self) == str(o)

    def __ne__(self, o):
        return not (self == o)

    def __hash__(self):
        return hash(str(self))

    def apply(self, obj):
        position_args = []
        color_args = []
        for key in self._template_args:
            if self._template_args[key].__class__.__name__ == "PositionDescription":
                position_args.append(self._template_args[key])
            elif self._template_args[key].__class__.__name__ == "ColorDescription":
                color_args.append(self._template_args[key])
            else:
                raise ValueError

        position_ext = obj.get_all_positions()
        for position_arg in position_args:
            position_ext = position_ext & position_arg.get_extension()

        if len(position_ext) == 0:
            return 0.0

        full_meaning = 1.0
        for color_arg in color_args:
            meaning = 0.0
            for coordinate in position_ext:
                meaning = max(meaning, color_arg.apply(obj.get_shape(coordinate).get_color()))
            full_meaning *= meaning

        return full_meaning


class PositionDescription:
    def __init__(self, desc_str, extension):
        self._desc_str = desc_str
        self._extension = extension # Set of coordinates

    def __str__(self):
        return self._desc_str

    def get_extension(self):
        return self._extension

class ColorDescription:
    def __init__(self, desc_indices):
        self._desc_indices = desc_indices
        self._desc_str = " ".join([D_COLOR["utterance"].get_feature_token(desc_indices[0][j,0]).get_value() for j in range(1,desc_indices[1][0]-1)])
        self._desc_str = self._desc_str.replace(" -er", "er").replace(" -est", "est").replace(" -ish", "ish")
        self._desc_str = self._desc_str.replace(" ?", "").replace(" .", "").replace(" !", "").replace(" ,", "")

    def __str__(self):
        return self._desc_str

    def apply(self, color):
        # Utterance, world , observation
        return MODEL_COLOR_MEANING((Variable(self._desc_indices[0].transpose(0,1).unsqueeze(0)), self._desc_indices[1].unsqueeze(0)), Variable(torch.zeros(1,1).long()), Variable(color.get_cielab_tensor())).data[0,0,0]

class Color:
    def __init__(self, hsl_obj):
        self._hsl_obj = hsl_obj
        self._cielab_arr = self._hsl_to_cielab(hsl_obj)

    def _hsl_to_cielab(self, hsl_obj):
        hsl = [hsl_obj["H"], hsl_obj["S"], hsl_obj["L"]]
        rgb = np.array(hsls_to_rgbs([map(int, hsl)]))[0]
        lab = np.array(rgbs_to_labs([rgb]))[0]
        return lab

    def get_cielab_tensor(self):
        return torch.from_numpy(self._cielab_arr).unsqueeze(0).float()

    def sample_description(self):
        desc_indices = MODEL_COLOR_S0.sample(input=self.get_cielab_tensor(), max_length=MAX_UTTERANCE_LENGTH)[0]
        desc = ColorDescription(desc_indices)

        while "#unc#" in str(desc) or len(str(desc).split(" ")) > 7:
            desc_indices = MODEL_COLOR_S0.sample(input=self.get_cielab_tensor(), max_length=MAX_UTTERANCE_LENGTH)[0]
            desc = ColorDescription(desc_indices)

        return desc

class Shape:
    def __init__(self, color, row, column, obj_width, obj_height):
        self._color = color
        self._row = row
        self._column = column
        self._obj_width = obj_width
        self._obj_height = obj_height

    def get_color(self):
        return self._color

    def get_position(self):
        return (self._row, self._column)

    def is_center(self):
        return (not self.is_column_left()) and (not self.is_column_right()) and \
           (not self.is_row_top()) and (not self.is_row_bottom())

    def is_corner(self):
        return (self._row == 0 or self._row == self._obj_height - 1) and \
            (self._column == 0 or self._column == self._obj_width - 1)

    def is_column_left(self):
        return self._column == 0

    def is_column_right(self):
        return self._column == self._obj_width - 1

    def is_row_top(self):
        return self._row == 0

    def is_row_bottom(self):
        return self._row == self._obj_height - 1

    def sample_consistent_template(self, utt_templates):
        template = np.random.choice(utt_templates)
        while template.has_corner_arg() and not self.is_corner():
            template = np.random.choice(utt_templates)
        return template

    def sample_column_description(self):
        if self.is_column_left():
            ext = set([])
            for i in range(grid_dim):
                ext.add((i,0))
            return PositionDescription(np.random.choice(COLUMN_LEFT_DESCRIPTIONS), ext)
        elif self.is_column_right():
            ext = set([])
            for i in range(grid_dim):
                ext.add((i,grid_dim-1))
            return PositionDescription(np.random.choice(COLUMN_RIGHT_DESCRIPTIONS), ext)
        else:
            ext = set([])
            for i in range(grid_dim):
                for j in range(1, grid_dim-1):
                    ext.add((i,j))
            return PositionDescription(np.random.choice(COLUMN_MIDDLE_DESCRIPTIONS), ext)

    def sample_row_description(self):
        if self.is_row_top():
            ext = set([])
            for i in range(grid_dim):
                ext.add((0,i))
            return PositionDescription(np.random.choice(ROW_TOP_DESCRIPTIONS), ext)
        elif self.is_row_bottom():
            ext = set([])
            for i in range(grid_dim):
                ext.add((grid_dim-1,i))
            return PositionDescription(np.random.choice(ROW_BOTTOM_DESCRIPTIONS), ext)
        else:
            ext = set([])
            for i in range(grid_dim):
                for j in range(1, grid_dim-1):
                    ext.add((j,i))
            return PositionDescription(np.random.choice(ROW_MIDDLE_DESCRIPTIONS), ext)

    def sample_position_description(self):
        row_desc = self.sample_row_description()
        col_desc = self.sample_column_description()
        ext = row_desc.get_extension() & col_desc.get_extension()

        desc_str = str(row_desc)
        if not self.is_center():
            desc_str += " " + str(col_desc)

        return PositionDescription(desc_str, ext)

class GridObject:
    def __init__(self, shapes, target):
        self._shapes = shapes
        self._target = target

        self._shapes_dict = dict()
        for shape in self._shapes:
            self._shapes_dict[shape.get_position()] = shape

    def get_all_positions(self):
        return ALL_POSITIONS

    def get_shapes(self):
        return self._shapes

    def get_shape(self, coordinate):
        return self._shapes_dict[coordinate]

    def is_target(self):
        return self._target

    @staticmethod
    def make_from_state(state_json):
        objs_dict = dict()
        for key in state_json:
            if not key.startswith("sObj"):
                continue
            key_parts = key.split("_")
            obj_idx = int(key_parts[0][4:])
            shp_idx = int(key_parts[1][3:])
            dim = key_parts[2][3:]
            if obj_idx not in objs_dict:
                objs_dict[obj_idx] = dict()
            if shp_idx not in objs_dict[obj_idx]:
                objs_dict[obj_idx][shp_idx] = dict()
            objs_dict[obj_idx][shp_idx][dim] = float(state_json[key])

        target_idx = int(state_json["sTarget"])
        objs = [None for i in range(len(objs_dict))]
        for i in range(len(objs_dict)):
            shapes = [None for j in range(len(objs_dict[i]))]
            for j in range(len(objs_dict[i])):
                shapes[j] = Shape(Color(objs_dict[i][j]), j / OBJECT_WIDTH, j % OBJECT_WIDTH, OBJECT_WIDTH, OBJECT_HEIGHT)
            objs[i] = GridObject(shapes, target_idx == i)
        return objs

    @staticmethod
    def get_target(objs):
        for i in range(len(objs)):
            if objs[i].is_target():
                return objs[i]
        return None

def make_utt_prior_support(objs, utt_templates):
    target_obj = GridObject.get_target(objs)
    utts = []
    for shape in target_obj.get_shapes():
        for i in range(UTTERANCE_PRIOR_SAMPLES_PER_COLOR):
            template = shape.sample_consistent_template(utt_templates)
            template_args = template.sample_args(shape)
            utts.append(Utterance(template, template_args))
    return utts

def sample_utterance(round_json, utt_templates):
    state_json = round_json["events"][0]
    objs = GridObject.make_from_state(state_json)
    target_obj = GridObject.get_target(objs)
    utts = make_utt_prior_support(objs, utt_templates)
    p_l0 = np.zeros(shape=(len(utts))) # p(t|u) for each u
    for i in range(len(utts)):
        norm_l0 = 0.0
        for j in range(len(objs)):
            norm_l0 += utts[i].apply(objs[j])
        p_l0[i] = utts[i].apply(target_obj) / norm_l0
    p_s1 = (p_l0 ** SPEAKER_ALPHA)/np.sum(p_l0 ** SPEAKER_ALPHA)
    utt = np.random.choice(utts, p=p_s1)
    return str(utt)

if grid_dim == 1:
    utt_templates = [UtteranceTemplate(ARG_COLOR)]
else:
    utt_templates = [UtteranceTemplate(ARG_COLOR + " in " + ARG_POSITION_CORNER + " corner"), \
                 UtteranceTemplate(ARG_COLOR + " color in " + ARG_POSITION_CORNER + " corner"), \
                 UtteranceTemplate(ARG_COLOR + " in " + ARG_POSITION_ROW + " row"), \
                 UtteranceTemplate(ARG_COLOR + " in " + ARG_POSITION + " spot"), \
                 UtteranceTemplate(ARG_COLOR + " on " + ARG_POSITION_ROW + " row"), \
                 UtteranceTemplate(ARG_COLOR + " in the " + ARG_POSITION), \
                 UtteranceTemplate(ARG_COLOR + " " + ARG_POSITION), \
                 UtteranceTemplate(ARG_COLOR + " " + ARG_POSITION_CORNER + " corner"), \
                 UtteranceTemplate(ARG_COLOR + " on the " + ARG_POSITION_ROW + " row") \
                ]

D = DataSet.load(trials_dir, id_key="gameid")
g = 1
for datum in D:
    game_json = datum.to_dict()

    for i in range(len(game_json["records"])):
        round_json = game_json["records"][i]
        utt = sample_utterance(round_json, utt_templates)
        round_json["events"].append({
          "type": "Utterance",
          "eventType" : "utterance",
          "sender": "speaker",
          "contents": utt,
          "time": round_json["events"][0]["time"] + 1
        });

        #round_json["events"].append({
        #  "type": "ActionGrid",
        #  "eventType" : "action",
        #  "time": round_json["events"][0]["time"] + 2
        #}); # NOTE: Filler action

        print "Utterance for game " + str(g) + " round " + str(i) + ": " + utt

    g += 1

    with open(join(output_dir, str(game_json["gameid"])), 'w') as fp:
        json.dump(game_json, fp)
