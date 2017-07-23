# ltprg

This repository contains code for *learning to play reference games*.  The
repository is organized into an *examples* directory containing data examples,
a *src/main* directory containing classes and functions for training and
evaluating models, and a *src/test* directory containing tests, experiments,
and scripts that run the model training and evaluation functions.

The library relies on the [mungpy](https://github.com/forkunited/mungpy)
repository at for munging the reference game data into a common format that
can be featurized and used by the models.  Some
instructions for how to push reference game data through this pipeline are
given below.  The main steps in this pipeline described below include:

1. [Converting game data to the JSON format](#converting)
2. [Extracting a state-utterance-action data set](#extracting-sua)
3. [Computing and saving feature matrices from the data](#featurization)
4. [Reloading saved feature matrices into memory](#loading)

## Converting game data to the JSON format

Pushing the data from all the reference games into a standard format will
allow for easy re-use of featurization and modeling code.  Currently, we use
the format shown in the JSON schema shown below, and for which there are
more examples from the color data set in
[examples/games/json/color](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color).

    ```json
    {
     "gameid" : "*unique game identifier string*",
     "records": [{ "roundNum": 1,
                   "events": [
                       { "type": "*StateSubtype*", "time": 1476996301986, "..." : "..." },
                       { "type": "Utterance", "time": 1476996265147, "sender": "speaker", "contents": "*Stuff said by speaker*"},
                       { "type": "Utterance", "time": 1476996265180, "sender": "listener", "contents": "*Stuff said by listener*"},
                       { "type": "Utterance", "time": 1476996265190, "sender": "speaker", "contents": "*More stuff said by speaker*"},
                       { "...", "..."},
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
field that contains a list of numbered game round objects.  Each round object
consists of a *"roundNum"* (round number) and a list of events.  Each event
is either a state, an utterance, or an action.  Each of these events can
have several game-specific dimensions with arbitrarily complicated substructure
determined by the dimensions of the events of a particular game.

There are two options for converting reference game data into this format.  The
first is to write a custom script to convert from your source format to the JSON
format.  The second option is to put your data into an alternative CSV
representation, and then use
[test/py/ltprg/data/make_game_csv_to_json.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_game_csv_to_json.py)
to convert to the JSON format.  See the documentation at the top of that script
for details about the CSV format.

### Producing NLP annotations for JSON game data

FIXME

## Extracting a state-utterance-action data set

FIXME

## Computing and saving feature matrices from the data

FIXME

## Reloading saved feature matrices into memory

FIXME
