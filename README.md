# NER-Evaluation
Full Named-Entity evaluation metrics

This repository contains implementation of Named-Entity Recognition evaluation metrics as defined by the SemEval 2013 9.1 task.

You can find the blog post associated to this code here:

* http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

coverage run --rcfile=setup.cfg --source=ner_evaluation/ -m pytest
coverage report