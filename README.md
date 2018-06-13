# ltprg

This repository contains code for learning to play reference games.  
The code relies on the [mungpy](https://github.com/forkunited/mungpy)
repository for generic utilities to munge the reference game data into a JSON 
format, vectorize this JSON into several tensor views of the data, and run 
training/evaluations of the learning models using this tensor data.   

The remainder of this README is organized into the following sections:

* [A summary of the contents of this repository](#summary-of-repository-content)
* [Details of how to setup the code environment](#setup-instructions)
* [Instructions for pre-processing reference game data](#how-to-preprocess-reference-game-data-for-modeling)
* [Instructions for training and evaluating models](#how-to-train-and-evaluate-models)

## Summary of repository contents

The repository is organized into a *config* directory containing configuration
files for modeling experiments, a *scripts* directory containing templates for 
shell scripts for running the tasks in *src/test*, an *examples* directory 
containing data examples, a *src/main* directory containing a library of classes 
and functions for training and evaluating models, and a *src/test* directory containing tests, 
experiments, and scripts that call model training and evaluation functions.
This separation between the main libraries in *src/main* and the 
scripts/tests in *src/test* was inspired by the layout used in Java 
Maven projects, and might seem annoyingly over-complicated---leading
to unnecessarily deep directory structures that aren't very typical of 
Python projects.  It is true that this is annoying, but the structure has also had the 
benefit of keeping the one-off scripty type things separated away from 
the main library code in *src/main*. 

### Script templates

After being filled in with local paths, the shell script templates in *scripts*
can be run to call various Python scripts in *src/test* with the configurations in
*config* to preprocess the data and train various models.  The details of the steps for
the pre-processing scripts are given in the the [data preprocessing](#how-to-preprocess-reference-game-data-for-modeling)
section, and the details on the model training and evaluation are given in the
[modeling](#how-to-train-and-evaluate-models) section.

### Configuration files

The *config* directory contains configuration files stored in JSON format which are used 
for setting hyper-parameters within the Python model training and evaluation 
scripts (e.g. in *src/test/py/ltprg/game/colorGrids/model/learn_RSA.py*).  These configurations
are currently (as of May 2018) split into the following subdirectories of *config*:

* *game/colorGrids/data* - Specifications for feature sets, data, and subsets of the
data to use within experiments.

* *game/colorGrids/eval* - Specifications for evaluations (log-likelihood, accuracy, etc) to use
in evaluating models.

* *game/colorGrids/learn* - Specifications for hyper-parameters of the learning algorithm (learning rate,
batch size, etc) to use during training

* *game/colorGrids/model* - Specifications for model hyper-parameters (e.g. RSA speaker 
rationality, number of units in hidden layers, etc)

* *game/colorGrids/old* - Old specifications that are no longer used, but kept around in case
they might be helpful for reference in the future

See the section below on [modeling](#how-to-train-and-evaluate-models) for details on how these 
are used.

### Data subdirectories

The data in *examples* is organized into the following directories:

* *games/csv/* - Source csv files from mturk containing reference game data for various games

* *games/json/* - Games converted to the JSON format described in the section below on
preprocessing the game data.  This is the form of the data used by the rest of the 
featurization/modeling code.

* *games/misc/* - Miscellaneous, relatively unimportant junk

* *games/splits/* - Files describing partitions of the data sets into train/dev/test 
partitions.

Note that the *colorGrids* sub-directories under the above (csv, json, etc) contain most 
of the relevant game data that
is used as of May 2018.  These contain the recently collected color grids from mturk, and also
merged with the colors data set from Monroe et al (2017).  
In *games/json/colorGrids*, there are json versions of the color grid data collected from mturk 
(sanitized and annotated under *clean_nlp*), and also merged with
the color data from Monroe et al (2017) color data (under *merged*).  
Directories named *sua_speaker* contain the state-utterance-action 
format of the data described in the [data pre-processing](#how-to-preprocess-reference-game-data-for-modeling) sections of the README below.  
The full merged data set used for training all color grid and color RSA models 
(as of May 2018) is in *games/json/colorGrids/merged/sua_speaker*.
The split for all this data is in *games/splits/colorGrids_merged*
(which represents a 34/33/33 train/dev/test split of the original colors data merged with a  
80/10/10 train/dev/test split of the color grid data).

### Python modules

The Python libraries in *src/main/py/ltprg* are organized into the following modules:

#### Experiment configuration parsing 

* *ltprg.config.rsa* - Functions for parsing configuration files (in *config*) into 
PyTorch RSA modules and evaluations (from *ltprg.model.rsa*)

* *ltprg.config.seq* - Functions for parsing configuration files (in *config*) into
PyTorch sequence model modules (from *ltprg.model.seq*)

#### Data manipulation and pre-processing

* *ltprg.data.curriculum* - Functions for re-ordering data sets according to 
training curricula (e.g. ordering of game rounds according to the lengths of speaker 
utterances)

* *ltprg.data.feature_helpers* - Helper functions for computing vectorized views
of the reference game JSON data.  These vectorized views are used as inputs to the 
learning models.

* *ltprg.data.feature* - Feature classes used by *ltprg.data.feature_helpers* to compute
vectorized views of the reference game JSON data.

#### Modules specific to particular reference games

* *ltprg.game.color.properties* - Some helper functions for manipulating data from the
color reference game (this was imported from another library, and was never fully 
integrated into this one, but a few of the functions here are used by the data 
preprocessing code). 

* *ltprg.game.color.eval* - Model evaluations specific to the color reference game 
(e.g. for outputting visualizations of learned meaning functions computed over 
*Hue x Saturation* color space)

* *ltprg.game.color.util* - Utilities specific to color games

#### Model components

* *ltprg.model.dist* - Represenations of probability distributions

* *ltprg.model.meaning* - Modules for computing RSA meaning functions

* *ltprg.model.obs* - Modules to compute over observations within RSA
when prior to conditioning the world and utterance priors (these are used within
*ltprg.model.rsa* as *observation_fns*, especially for SNLI RSA models and others
that condition the priors on embeddings from observed sequences)

* *ltprg.model.prior* - Modules for computing utterance and world priors within
RSA models

* *ltprg.model.rsa* - RSA modeling and evaluation modules

* *ltprg.model.seq_heuristic* - Heuristics to guide sequence model sampling and 
search procedures (e.g. used for sampling utterance prior supports that contain 
utterances which score highly according to literal listeners)

* *ltprg.model.seq* - Sequence model modules

#### Miscellaneous utilities

* *ltprg.util.file* - Utilities for managing file I/O

* *ltprg.util.img* - Utitilities for manipulating images

### Python scripts

Some scripts in *src/test/py* are:

#### Data manipulation and pre-processing

* *ltprg/data/make_game_csv_to_json.py* - Take games from mturk in csv 
format and convert them to JSON objects (see *examples/games/csv* and 
*examples/games/json* for examples)

* *ltprg/data/annotate_json_nlp.py* - Process utterances from
games (in JSON format) to produce annotations with NLP tools (e.g. tokenize, 
lemmatize, etc), and output new annotated games in JSON format

* *ltprg/data/make_sua.py* - Transform reference games in JSON format (with
one JSON object per game) into a data set with one state-utterance-action JSON 
object per round (representing single training examples for learning models)

* *ltprg/data/compute_utt_length_distribution.py* - Compute distribution
of utterance lengths across games

* *ltprg/data/make_wv_feature_lookup.py* - Builds and save a numpy matrix from 
word vectors stored in a feature set (this was useful for initializing sequence 
models with GLoVe embeddings for SNLI)

#### Scripts specific to color grid (and merged colors) reference games

The *ltprg/game/colorGrids* directory contains Python scripts particular to the 
color grids data set (and also the colors data set that has been converted
and merged into this same format).  There are older scripts specific to other 
reference games in other subdirectories of *ltprg/game*, but currently 
(as of May 2018), the only actively used scripts are the *colorGrids* ones.
They are as follows:

##### Data pre-processing

* *ltprg/game/colorGrids/data/annotate_with_targets.py* - Annotate the data 
with explicit representations of the target objects for each round that can
be easily featurized (this script is a bit of a hack).  Data that has been
run through this script contains *targetObj* fields containing representations
of the target objects.

* *ltprg/game/colorGrids/data/compute_stats.py* - Compute various statistics
over the data (e.g. human listener accuracy)

* *ltprg/game/colorGrids/data/convert_from_color.py* - Convert the Monroe et
al (2017) color data set into the same format as the color grid data.

* *ltprg/game/colorGrids/data/featurize_sua.py* - Compute vectorized 
representations of the data for use within learning models.

* *ltprg/game/colorGrids/data/make_syn_split.py* - Make train/dev/test 
partitions of the data

* *ltprg/game/colorGrids/data/merge_data.py* - Merge various data sets into a
single data set (e.g. merge the Monroe et al. (2017) color data with the
newer color grids data)

##### Modeling

* *ltprg/game/colorGrids/model/learn_RSA.py* - Train RSA models

* *ltprg/game/colorGrids/model/learn_SRSA.py* - Train RSA models that use
the same sequence model for both their meaning functions and utterance 
priors (this was an idea pursued briefly, and may be worthwhile to look
at again in the future)

* *ltprg/game/colorGrids/model/learn_S.py* - Train sequence models (i.e.
vanilla language models and S0 models)

* *ltprg/game/colorGrids/model/test_prior.py* - Compute examples of 
utterance priors under various settings to visualized and evaluate 
qualitatively

#### Results post-processing

* *ltprg/results/aggregate.py* - Take result files output by several 
training/evaluation jobs (e.g. under different seeds and hyper-parameter 
settings with a script like *ltprg/game/colorGrids/model/learn_RSA.py*), 
and merge them into a single tsv file.

#### Visualization and investigation of learned meaning functions

* *ltprg/meaning/* - This directory will eventually contain several 
scripts and tools for investigating learned meanings

#### Tests

* *ltprg/model/test_rsa.py* - Contains a test for the RSA PyTorch module

### Other stuff (e.g. game visualization)

The directory *src/test/html/viewData/* contains web pages used to visualize the 
data set.  Specifically, the color and colorGrids data, and also utterance 
priors output by *ltprg/game/colorGrids/model/test_prior.py* can be visualized 
using *src/test/html/viewData/colorGrids/view.html* (by just opening this file in a
browser, and using the form to select a JSON file representing a game or 
the utterance prior output)

## Setup instructions

To setup, first clone the repository:

    git clone https://github.com/forkunited/ltprg

You will also need to clone the mungpy repository that contains several data
munging and experiment utilities upon which ltprg relies:

    git clone https://github.com/forkunited/mungpy

Next, set your PYTHONPATH environment variable to include the paths to the
Python libraries within these cloned repositories:

    export PYTHONPATH=$PYTHONPATH:/path/to/ltprg/src/main/py/:/path/to/mungpy/src/main/py/

The code generally relies on PyTorch for training models, so install 
that if it is not already installed.  Currently, PyTorch version 0.3.0.post4 
is used.  You can download it at https://pytorch.org/previous-versions/.

## How to preprocess reference game data for modeling

Before running modeling experiments, it's useful to push all the reference
game data into a standard format that can be manipuated by models through a
common set of data structures.  This section describes how to push CSV
data generated by mturk experiments into a more easily manipulable JSON format,
and then generate vectorized views of this data that are used by the learning
models.  The main steps in pipeline include:

1. [Converting CSV game data to the JSON format](#converting-game-data-to-the-json-format)
2. [Producing NLP annotations for the JSON data](#producing-nlp-annotations-for-json-game-data)
3. [Extracting state-utterance-action data sets from JSON data](#extracting-state-utterance-action-data-sets)
4. [Partitioning data into train-dev-test splits](#partitioning-data-for-training-and-evaluation)
5. [Computing and saving feature matrices from the data](#computing-and-saving-feature-matrices-from-the-data)
6. [Reloading saved feature matrices into memory](#reloading-saved-feature-matrices-into-memory)

Steps 1 though 5 can be performed just once for a new reference game data set,
and step 6 is used in experiments to load the processed data for modeling, etc. 
A template for a single script that performs steps 1 through 5 is given in 
[scripts/preprocess.sh](https://github.com/forkunited/ltprg/blob/master/scripts/preprocess.sh).  
To setup the pipeline, this script template should be copied and filled in with
paths to data that are specific to the local environment.  Each of the pre-processing
steps is described in some detail below.

(Note that the color and color grids data sets under *examples/games/json/colorGrids* have been
generated through additional steps to the pipeline described here.  These additional
steps were necessary for merging the color and color grids data into a single format,
and they can be pefromed using the color-grid preprocessing script template at 
[scripts/preprocess_cg.sh](https://github.com/forkunited/ltprg/blob/master/scripts/preprocess_cg.sh)
In general, when new reference game data comes through a single CSV format, it's likely that
the steps described below, and performed by 
[scripts/preprocess.sh](https://github.com/forkunited/ltprg/blob/master/scripts/preprocess.sh)
should be at least nearly sufficient for pre-processing.)

### Converting game data to the JSON format

Pushing the data from all the reference games into a standard format 
allows for easy re-use of featurization and modeling code.  Currently, we use
the format shown in the JSON schema below.  There are 
more examples of this format from the color data set in
[examples/games/json/color](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color).

```json
{
 "gameid" : "*unique game identifier string*",
 "records": [{ "roundNum": 1,
               "events": [
                   { "type": "*StateSubtype*", "time": 1476996301986, "..." : "..." },
                   { "type": "Utterance", "time": 1476996265147, "sender": "speaker",
                     "contents": "*Stuff said by speaker*"
                   },
                   { "type": "Utterance", "time": 1476996265180, "sender": "listener",
                     "contents": "*Stuff said by listener*"
                   },
                   { "type": "Utterance", "time": 1476996265190, "sender": "speaker",
                     "contents": "*More stuff said by speaker*"
                   },
                   { "..." : "..." },
                   { "type": "*ActionSubtype*", "time": 1476996267239, "..." : "..." }
               ]
             },
             { "roundNum": 2, "events": [ { "...": "..." } ]},
             { "roundNum": 3, "events": [ { "...": "..." } ]},
             { "..." : "..."}
           ]
}
```

Note that in the above schema, place-holder values are given between the
asterisks, and the "..." fields indicate that the object could contain more
fields.

In this format, each reference game is represented by a single JSON object
containing a *"gameid"* unique identifier field for the game, and a *"records"*
field that contains a list of numbered game-round objects.  Each round object
consists of a *"roundNum"* (round number) and a list of events.  Each event
is either a state, an utterance, or an action.  Each of these events can
have several game-specific dimensions with arbitrarily complicated substructure
determined by the dimensions of the events of a particular game.

There are two options for converting reference game data into this format.  The
first is to write a custom script to convert from your source format to the JSON
format.  Alternatively, if you have a CSV representation of your data, then you
can use
[test/py/ltprg/data/make_game_csv_to_json.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_game_csv_to_json.py)
to convert to the JSON format.  See the documentation at the top of that script
for details about the CSV format that this script expects.

### Producing NLP annotations for JSON game data

When the reference game data is in JSON format, the utterances from the game can
be pushed through the Stanford CoreNLP pipeline using the script at
[test/py/ltprg/data/annotate_json_nlp.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/annotate_json_nlp.py).
See the documentation at the top of that script for details.  Also, there are 
examples of NLP annotated JSON color data in
[examples/games/json/color_nlp](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color_nlp)
(each line of each file contains a JSON object representing a game).

### Extracting state-utterance-action data sets

Several reference game learning models depend on training from examples
that consist of a single round represented by a game state, speaker utterances, 
and a listener action.  For example, training
the RSA listener models to play the color reference game depends on having
one example per game round---with a state of three colors, 
the speaker utterances, and either the target color referred to by the speaker or the
color clicked on by the listener.  The script at
[test/py/ltprg/data/make_sua.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_sua.py)
constructs state-utterance-action examples like this from the JSON game data.  Also, there
are examples of state-utterance-action data for the color data set in
[examples/games/json/color_sua_speaker](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color_sua_speaker).

### Partitioning data for training and evaluation

When training and evaluating models, it's useful to partition the data into 
train/dev/test sets---or other splits.  For this purpose, the
[mung.data.Partition](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/data.py)
class is useful for loading partitions of the data into memory, and splitting
data sets and feature matrices according to these partitions.  The class assumes
that the partitions are stored in the following JSON format:

```json
{
  "size": 948,
  "parts": {
    "train": { "1124-1": 1, "8235-6": 1, "..." : 1 },
    "dev": { "2641-2": 1, "4235-3": 1, "..." : 1 },
    "test": { "5913-4": 1, "1212-5": 1, "..." : 1 }
  }
}
```

The *"size"* field specifies the number of elements across all parts of the
partition, and the *"parts"* field contains the parts of the partition.  
Each part contains all
of the keys representing objects in the partition.  In the example shown above,
each key is a game ID.  The *mung.data.Partition*
can *split* a data set according to this partition and a "key function" that
maps datums from a data set to the keys in the partition.  Since the keys are
game IDs in the above example, the "key function" would need to map datums to
their game IDs to split a data set.

The script at 
[test/py/ltprg/data/make_split.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_split.py)
takes a path to a set of JSON game data, and uses mung.data.Partition to create a 
split based on game ID. There are many examples of data splits in
[examples/games/splits](https://github.com/forkunited/ltprg/tree/master/examples/games/splits).

### Computing and saving feature matrices from the data

Feature matrices can be computed from the state-utterance-action and game
data sets in the JSON format described above.  In particular, the
[mungpy.feature_helpers](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/feature_helpers.py)
module (from
[mungpy](https://github.com/forkunited/mungpy)) contains functions for
constructing feature matrices and sequences of feature
matrices (e.g. representing utterances).   The features in the matrices can
represent values stored in the JSON data examples.  These values can be either
numerical values stored directly in properties of the JSON objects, or tokens from an
enumerable vocabulary (e.g. of utterance words/lemmas).  The following
functions from *mung.feature_helpers* are used to construct the matrices:

* *featurize_path_enum* : This constructs a feature matrix where each column  
represents a token from some enumerable vocabulary extracted from the data, and
each row represents the values of the features for a particular datum.  So, the
entry at row *i* and column *j* indicates whether token *j* occurs in datum *i*.  
As a more compact alternative to this sparse one-hot representation, this function
can also store the token index numbers directly in the matrix.  Prior to
constructing the matrix, this function also constructs the token vocabulary
by gathering all possible tokens from across the data set.

* *featurize_path_enum_seqs* : This works similar to *featurize_path_enum*,
except that it computes a sequence of matrices rather than a single matrix,
where matrix *i* contains a representation of token *i* in the sequence for each
datum.  If some datum sequences contain fewer elements than others, then they
are padded.  Currently, such matrix sequences are used to represent the sequences
of words in utterances across a data set.

* *featurize_path_scalars* : This constructs a feature matrix where each column
represents some numerical dimension from the examples in the data set.  

As an example of how these functions are used, consider the state-utterance-action
datums from the color data set.  Each of these datums has the following form:

```json
{
  "id": "6655-7_1_0", "gameid": "6655-7", "sua": 0, "roundNum": 1,
  "state": {
    "type": "StateColor",
    "lH_0": "248", "lH_1": "294", "lH_2": "101",
    "lS_0": "31", "lS_1": "48", "lS_2": "77",
    "lL_0": "50", "lL_1": "50", "lL_2": "50",
    "..." : "..."
  },
  "utterances": [
    {
      "type": "Utterance",
      "sender": "speaker", "contents": "bright blue", "time": 1476989293766,
      "nlp": {
        "type": "CoreNLPAnnotations",
        "lemmas": { "lemmas": [ "bright", "blue" ], "..." : "..." },
        "tokens": "...", "sents": "...", "pos": "..."
      }
    }
  ],
  "action": {
    "type": "ActionColor",
    "time": 1476989295346,
    "lClicked_0": "1", "lClicked_1": "0", "lClicked_2": "0"
  }
}
```

Note that in this example,  there are indicators of which color the listener
clicked in the *"lClicked_i"* fields of the *"action"* sub-object.  Given datums in
this form stored at *input_data_dir*, the following function will compute a
feature matrix containing rows of these *"lClicked_i"* indicators for each datum:

```python
mung.feature_helpers.featurize_path_scalars(
    input_data_dir, # Source data set
    join(output_feature_dir, "listener_clicked"), # Output directory
    partition_file, # Data partition
    lambda d : d.get("gameid"), # Function that partitions the data
    "listener_clicked", # Name of the feature
    ["action.lClicked_0", "action.lClicked_1", "action.lClicked_2"], # JSON paths to feature values
    init_data="train") 
```

The first two arguments to the function specify the input and output locations
on disk.  The third and fourth argument specify the location of a file storing
a data partition and the "key function" for partitioning the data (see the
section on [partitions](#data-partitons)), and the final *init_data*
specifies the part of this partition on which to *initialize* the features
(this is necessary when the feature initialization depends on some property of
the data, but should not depend on the values from the test data). 
The "listener_clicked" argument just gives a name for the feature.  Finally,
the list of "action.lClicked_0", "action.lClicked_1", and "action.lClicked_2"
specifies the JSON paths within the data from which to gather the feature
values.  This will construct a matrix where each row represents a
state-utterance-action example, and each the columns represent values from
"action.lClicked_0", "action.lClicked_1", and "action.lClicked_2".

As another example, the following function computes sequences of feature
matrices representing utterances from the state-utterance action data:

```python
mung.feature_helpers.featurize_path_enum_seqs(
    input_data_dir, # Source data set
    join(output_feature_dir, "utterance_lemma_idx"), # Output directory
    partition_file, # Data partition
    lambda d : d.get("gameid"), # Function that partitions the data
    "utterance_lemmas_idx", # Name of the feature
    ["utterances[*].nlp.lemmas.lemmas"], # JSON path into data examples
    15, # Maximum utterance length
    token_fn=lambda x : x.lower(), # Function applied to tokens to construct the vocabulary
    indices=True, # Indicates that indices will be computed instead of one-hot vectors
    init_data="train") 
```

The function first computes a vocabulary of lower-cased lemmas gathered from the
"utterances[\*].nlp.lemmas.lemmas" JSON path across the data set.  Then, it computes
a sequence of *15* padded feature matrices representing the lemmas of utterance 
tokens for each datum.

See
[test/py/ltprg/game/color/data/feature_sua.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/color/data/featurize_sua.py)
for some further examples of how feature matrices are constructed from the color
reference game data. Also see 
[test/py/ltprg/game/colorGrids/data/feature_sua.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/data/featurize_sua.py)
for a similar script that featurizes the more recent merged color/color-grid data. 

### Reloading saved feature matrices into memory

Assuming a state-utterance-action data set in *sua_data_dir* has been
featurized using the methods described in the previous section, the feature
matrices can be reloaded into memory from *features_dir* using code like the
following:

```python
D = MultiviewDataSet.load(
    sua_data_dir,
    dfmat_paths={
        "listener_clicked" : join(features_dir, "listener_clicked"),
        "listener_colors" : join(features_dir, "listener_colors"),
        "speaker_colors" : join(features_dir, "speaker_colors"),
        "speaker_observed" : join(features_dir, "speaker_observed"),
        "speaker_target_color" : join(features_dir, "speaker_target_color"),
        "speaker_target" : join(features_dir, "speaker_target")
    },
    dfmatseq_paths={
        "utterance_lemma_idx" : join(features_dir, "utterance_lemma_idx")
    })
partition = Partition.load(partition_file)
D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]

batch = D_train.get_random_batch(5)
```

This loads feature matrices from each of the sub-directories under *features_dir*
as specified by the *dfmat_paths* argument, and feature matrix sequences from
the paths specified by the *dfmatseq_paths* argument.  The data is then split
according to the partition loaded from *partition_file* (see the
section on [partitions](#data-partitons)).  Sub-matrices representing a random
batch of 5 examples are extracted using the *get_random_batch* method on the
last line.  The returned *batch* object is a dictionary with keys for each
of the loaded feature matrices and feature matrix sequences.  So for example,
*batch["listener_clicked"]* will contain a matrix with 5 rows, and columns
representing where the listener clicked (as computed in the previous
  section).

See the unit tests at
[test/py/ltprg/game/color/data/validate_data.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/color/data/validate_data.py)
for more examples of how to access the data.  

## How to train and evaluate models

After the data has been preprocessed according to the steps described
above (and given in the script template *scripts/preprocess.sh*), we can
train and evaluate models on the resulting vectorized data sets.  This 
training/evaluation can be run using scripts based on the templates in
*scripts*, which train RSA listeners and language models based on the 
color and color grid subsets of the merged data in 
*examples/games/json/colorGrids/merged/sua_speaker*.  These shell script 
templates call the Python scripts in 
*src/test/py/ltprg/game/colorGrids/model/*.  Each Python script takes a 
set of configuration files from *config*, and trains/evaluates a single model,
logging the results to some directory.  The first three subsections below give 
the high level details of this process.  

1. [Configuring experiments](#configuring-experiments)
2. [Training and evaluating models](#model-training-and-evaluation)
3. [Understanding evaluation output](#training-and-evaluation-output) 

The fourth and fifth sections give
lower level details of the design for training RSA and sequence models:

4. [Design for RSA models](#design-for-rsa-models)
5. [Design for sequence models](#design-for-sequence-models)

### Configuring experiments

The Python training and evaluation scripts (e.g. for training RSA models in
[learn_RSA.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_RSA.py))
are configured through the JSON configuration files in the *config* directory.
These configuration files contain hyper-parameter settings and specifications for architectural 
details of the model.  As a concrete example, the [learn_RSA.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_RSA.py)
script loads data, model, learning, and evaluation 
configurations through the following lines:

```python
data_config = Config.load(args.data, environment=env)
model_config = Config.load(args.model, environment=env)
learn_config = Config.load(args.learn, environment=env)
train_evals_config = Config.load(args.train_evals, environment=env)
dev_evals_config = Config.load(args.dev_evals, environment=env)
test_evals_config = Config.load(args.test_evals, environment=env)
```

An example of a configuration that can be loaded into the *learn_config* line 
above is given in [cgmerged_src_color3_data.json](https://github.com/forkunited/ltprg/blob/master/config/game/colorGrids/learn/rsa/cgmerged_src_color3_data.json).  
This file contains the following JSON object, which specifies the parameters
for the training algorithm:

```json
{
    "max_evaluation" : true,
    "data" : "$!{train_data}",
    "data_size" : "$!{train_data_size}",
    "iterations" : 10000,
    "batch_size" : 128,
    "optimizer_type" : "ADAM",
    "learning_rate" : 0.005,
    "weight_decay" : 0.0,
    "gradient_clipping" : 5.0,
    "log_interval" : 100
}
```

Notice that the *data* and *data_size* fields in this object contain
*$!{train_data}* and *$!{train_data_size}* placeholders for the name of 
the data and the size of the subset to train on.  These placeholders are defined
through the local environment, which is specified either through 
command-line options (i.e. giving *--train_data_size 1000*
when calling the script) or through a local environment configuration file which
specifies local paths to data and other resources.   The file
[env.json](https://github.com/forkunited/ltprg/blob/master/env.json) gives a template
for this local environment configuration, and it should be copied to a local *env_local.json*, 
and filled in with paths specific to the local machine.  This separation between
the local environment and the configuration gives (1) a means by which to keep local 
information out of the repository, and (2) a way to re-use parameter settings that 
tend to stay the same while also offering the flexibility of specifying other 
parameters through the command line.  

The *data* field in the above configuration gives the name of the data subset to 
train on, where this data is defined through a data configuration file (e.g. see 
[here for example data configurations](https://github.com/forkunited/ltprg/blob/master/config/game/colorGrids/data)).
In general, the data configurations take some initial featurized data set, and 
split it into named subsets representing conditions that are useful for training 
and evaluation (e.g. *close*, *split*, and *far* conditions in the color data).

When setting up new experiments, you can get an idea of what configurations are 
necessary by looking through some of the examples in the [config](https://github.com/forkunited/ltprg/blob/master/config) 
directory, but there is also some documentation on what configuration fields are required
in the Python modules responsible for parsing the configurations.  The following
modules are responsible for parsing various types of configurations:

* [Data subsets](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/config/feature.py)
* [Learning algorithms](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/config/torch_ext/learn.py)
* [RSA models and evaluations](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/config/rsa.py)
* [Sequence models and evaluations](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/config/seq.py)

### Model training and evaluation

The Python training scripts (like 
[learn_RSA.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_RSA.py) 
and [learn_S.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_S.py))
configure the main training loop through the [training configuration module](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/config/torch_ext/learn.py).  
This module constructs a *Trainer* class from [learn.py](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/torch_ext/learn.py).
Among other arguments, the *Trainer* takes a "data parameter", loss criterion, evaluation metrics,
a model, and some data.  The trainer runs some SGD variant to train the model for a specified 
number of iterations while logging the evaluation results to a log file at regular intervals.
The trainer keeps track of the best evaluation of the model, and returns both the final model
from the full set of iterations, and the best model according to the main evaluation.

The trainer assumes that the model is a Python class that extends PyTorch's nn.Module,
and also has a *forward_batch* and *loss* methods.  The purpose of these methods is that 
they give a consistent signature that the *Trainer* can use across
many model types with different kinds of inputs and outputs. The *forward_batch* method should take
a data batch and a "data parameter", and compute a forward pass of the module on the batch,
indexing into it using the "data parameter".  The *loss* method should take a data batch,
a "data parameter", and a loss criterion, compute the forward pass, and then use the modules 
output to compute the loss according to the loss criterion.  Example implementations of
 *forward_batch* and *loss* methods are given in [seq.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/seq.py) 
 for sequence models and [rsa.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/rsa.py) 
 for RSA models.

The "data parameter" argument to *forward_batch* and *loss* specifies names of views within
the data batch that should be used by the model.  This gives a mapping between the types of
data that the model expects (e.g. sequences for sequence models) and their names within the 
given data set (e.g. utterances within reference games).  For example, there is a *DataParameter*
class specific to sequence models at the top of [seq.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/seq.py).
This class stores the name of the sequence view within the data, and the name of the non-sequential
input view within the data.  In the [model configuration file](https://github.com/forkunited/ltprg/blob/master/config/game/colorGrids/model/s0/attn_cgmerged.json) 
for specifying an S0 sequence model, these view names are given as "utterance" and "target_obj".  These
view names refer to the data views specified at the top of the 
[data configuration file](https://github.com/forkunited/ltprg/blob/master/config/game/colorGrids/data/cgmerged_cpos_unclean.json).
Every batch of data fed into *forward_batch* or *loss* by trainer will contain these vectorized views of the
data, indexed by the names "utterance" and "target_obj".

The *Trainer* class assumes that the given evaluations implement the *Evaluation* class in
[eval.py](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/torch_ext/eval.py).  An
evaluation should take a model, and output a number.  Several examples of existing 
evaluations (e.g. accuracy, loss according to some criterion, etc) are also given in 
[eval.py](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/torch_ext/eval.py).  There
also several RSA specific evaluations at the end of 
[rsa.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/rsa.py).

### Training and evaluation output

The Python training scripts,
[learn_RSA.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_RSA.py) 
and [learn_S.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_S.py)),
create new directories in which to store their output.  The output conists of:

* *log* - A log tsv file giving training evaluations at regular intervals
during training

* *config.json* - A JSON file storing all the configurations and arguments that were 
used to run the training.

* *model* - The stored trained model which can be reloaded into memory later for
further training or evaluation

* *results* - A tsv file containing a single line of final evaluations of the final
model on the dev set

* *test_results* - A tsv file containing a single line of final evaluation of the final 
model on the test set.  This is only output by the training scripts if explicitly specified 
(to avoid unnecessary test set evaluations)

Note that the final results in *result* and *test_results* just containing a single
line of evaluations of a single model.  Typically, it's useful to gather the results 
for many model trainings into a single tsv file.  The [aggregate.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/results/aggregate.py) script
is useful for aggregating the *results* files from many training runs into a single 
tsv (possibly averaging over some runs, like if running under the same hyper-parameters
with multiple random seeds).

Training RSA modules can also produce an evaluation for utterance priors which outputs
a directory containing JSON files storing the utterance prior supports at 
successive iterations of training for a subset of example contexts. These output 
priors can be visualized for the color grid and color games using web page visualization 
[src/test/html/viewData/colorGrids/view.html](https://github.com/forkunited/ltprg/blob/master/src/test/html/viewData/colorGrids/view.html).

### Design for RSA models

The RSA module [rsa.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/rsa.py) 
trained using the 
[learn_RSA.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_RSA.py) 
script produces listener RSA distributions (it can also produce speakers, but it has mostly been used
for listeners up until now (May 2018)).  A forward pass of the listener module (referred to briefly as "L" 
within the code) assumes that data examples contain listener "observations", speaker "utterances", and world 
"targets".  An observation consists of a context observable to the listener upon hearing 
the speaker's utterance, and the target is the referent that the listener should infer within that context.  
So, the listener's forward pass computes an RSA distribution over targets given observations and 
speaker utterances.  The computation is batched, and so the module takes observations of shape 
(Batch size x Observation), utterances of shape (Batch size x Utterance), and produces batches of 
distributions of shape (Batch size x Support size) stored in distribution batch objects from 
[dist.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/prior.py).  
Internally, the RSA distributions are computed using  
an utterance prior, a world prior, and a meaning function.  The RSA listener module in [rsa.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/rsa.py) 
computes these components using sub-modules from:

* [prior.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/prior.py) -
PyTorch sub-modules which take (Batch size x Observation) batches of observations, and produce 
(Batch size x Support size) prior distributions over utterances and worlds.  

* [meaning.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/meaning.py) -
PyTorch sub-modules which take (Batch size x Utterance prior support size) batches of utterance 
prior supports and (Batch size x World prior support size), and produce batches of meaning matrices
of shape (Batch size x Utterance prior support size x World prior support size). 

Many of the sub-modules in [prior.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/prior.py) 
produce distributions over sequences, and sub-modules in 
[meaning.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/meaning.py) compute
meanings over sequential utterance inputs.  The architectures for operating over these sequential 
inputs are defined by the SequenceModel sub-modules from 
[seq.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/prior.py).

Note that the listener RSA module is also overloaded to operate as an internal distribution within
the RSA recursion (e.g. as an L0 computed within an L1).  
This means that it can optionally take (Batch size x Utterance prior support size x Utterance) 
inputs from the utterance prior supports of a higher level pragmatic speaker (instead of the simpler top-level
(Batch size x Utterance) utterance batches), and produce (Batch size x Utterance prior support size x World distributions).
This makes the code a bit more difficult to understand, but if you're only interested in using the 
module (not editing it), you can ignore this detail, and treat the listener module as though it just overates 
over (Batch size x Utterance) batches and produces batches of world distributions.

Also, note that the RSA modules can take an optional "observation_fn" which computes some function
over observations before they are fed into the utterance and world prior modules.  This is especially useful 
when the observed context is a sequence (e.g. a premise sentence in SNLI), and it makes sense to compute 
embeddings of this sequnce to use as "worlds" in the supports of the world priors.  However, note that this 
functionality was tacked on in a hacky way, and is not completely implemented for the case of computing top-level
speaker distributions.

### Design for sequence models

The sequence modeling modules in [seq.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/seq.py) 
are used as sub-modules within RSA (for computing utterance priors and meaning functions), and also can be trained
on their own (e.g. for language modeling) using using the Python script 
 [learn_S.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/colorGrids/model/learn_S.py).
There is a generic *SequenceModel* PyTorch module defined near the top of [seq.py](https://github.com/forkunited/ltprg/blob/master/src/main/py/ltprg/model/seq.py), and below it there are 
several specific RNN extensions of this module.  The *SequenceModel* class implements several generic sequence 
model methods (e.g. for sampling, beam search, etc), and the extensions implement specific network architectures.

The sequence models generally assume that the input data examples consist of sequences (abbreviated *seq*) along with
 non-sequential inputs referred to as *input* (but some sequence models do not have this additional input).  The 
 sequences are represented as 
(Max sequence length x Batch size) tensors of indices into a token vocabulary or (Max sequence length x Batch size x Vector size)
tensors containing batches of sequences of vectors.  These sequences also come with a vector of size (Batch size) 
containing the sequence lengths, and the possible *inputs* have size (Batch size x Input vector size).
Given these inputs, the sequence models produce sequential output tensors of size (Sequence length x Batch size x Vector size) 
where the output "Vector size" is the same as the input vector (or token vocabulary) size.  