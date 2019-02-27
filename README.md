# Named Entity Evaluation as in SemEval 2013 task 9.1

My own implementation, with lots of input from [Matt Upson](https://github.com/ivyleavedtoadflax), of the Named-Entity Recognition evaluation metrics as defined by the SemEval 2013 - 9.1 task.

This evaluation metrics go belong a simple token/tag based schema, and consider diferent scenarios based on wether all the tokens that belong to a named entity were classified or not, and also wether the correct entity type was assigned.

You can find a more detailed explanation in the following blog post:

* http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/


## Notes:

In scenarios IV and VI the entity type of the `true` and `pred` does not match, in both cases we only scored against the true entity, not the predicted one. You can argue that the predicted entity could also be scored as spurious, but according to the definition of `spurious`:

* Spurius (SPU) : system produces a response which doesn’t exist in the golden annotation;

In this case it exists an annotation, but only with a different entity type, so we assume it's only incorrect


## Example:

You can see a working example on the following notebook:

- [example-full-named-entity-evaluation.ipynb](example-full-named-entity-evaluation.ipynb)

Note that in order to run that example you need to have installed:

- sklearn
- nltk
- sklearn_crfsuite

For testing you will need:

- pytest
- coverage

These dependencies can be installed by running `pip3 install -r requirements.txt`

## Code tests and tests coverage:

To run tests:

`coverage run --rcfile=setup.cfg -m pytest`

To produce a coverage report:

`coverage report`
