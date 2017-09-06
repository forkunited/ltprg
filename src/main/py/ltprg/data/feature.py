from mung.feature import FeatureType
# import sys
# sys.path.append('../../../../test/py/ltprg/game/color/properties/')
import numpy as np
import time
from ltprg.game.color.properties.alexnet import PartialAlexnet, rgb_to_alexnet_input
from ltprg.game.color.properties.colorspace_conversions import hsls_to_rgbs

class VisualEmbedding(FeatureType):
	def __init__(self, name, paths):
		FeatureType.__init__(self)
		# paths is a list of lists of HSLs, e.g. [[H1, S1, L1], [H2, S2, L2]]
		self._name = name
		self._paths = paths # e.g. [["state.sH_0", "state.sS_0", "state.sL_0"]]

	def get_name(self):
		return self._name

	def compute(self, datum, vec, start_index):
		output = []
		for color in self._paths:
			hsl = [datum.get(dim) for dim in color]
			if hsl == [None, None, None]:
				print hsl
				print datum.get("id")
			rgb = np.array(hsls_to_rgbs([map(int, hsl)]))[0]
			output.extend(self.cnn.forward(rgb_to_alexnet_input(rgb)
							).data.numpy()[0])
		vec[start_index : len(vec)] = output[start_index : len(output)]

	def get_size(self):
		return len(self.cnn.forward(rgb_to_alexnet_input([0, 0, 0])
				   ).data.numpy()[0]) * len(self._paths) # e.g. 4096 * 3

	def get_token(self, index):
		return None

	def __eq__(self, feature_type):
		if not isinstance(feature_type, VisualEmbedding):
			return False
		if self._name != feature_type.name:
			return False
		return True

	def init_start(self):
		self.cnn = PartialAlexnet('fc6')

	def init_datum(self, datum):
		pass

	def init_end(self):
		pass

	def save(self, file_path):
		obj = dict()
		obj["type"] = "VisualEmbedding"
		obj["name"] = self._name
		return obj
