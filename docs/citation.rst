Citation
========

If you use HAAM in your research, please cite our paper:

.. code-block:: bibtex

   @article{li2025dml,
     title={High-Dimensional Perception with the Double Machine Learning Lens Model 
            Equation (DML-LME)},
     author={Li, Raymond V. and Biesanz, Jeremy C.},
     journal={Psychometrika},
     year={2025},
     doi={10.xxxx/xxxxx}
   }

Software Citation
-----------------

To cite the HAAM software package specifically:

.. code-block:: bibtex

   @software{haam_package,
     title={HAAM: Human-AI Accuracy Model Python Package},
     author={Li, Raymond V. and Biesanz, Jeremy C.},
     year={2025},
     version={1.0},
     url={https://github.com/raymondli-me/haam}
   }

Related Papers
--------------

The HAAM framework builds on several foundational works:

**Lens Model Theory**

.. code-block:: bibtex

   @book{brunswik1952,
     title={The Conceptual Framework of Psychology},
     author={Brunswik, Egon},
     year={1952},
     publisher={University of Chicago Press}
   }

   @article{hammond1966,
     title={The psychology of Egon Brunswik},
     author={Hammond, Kenneth R and Stewart, Thomas R},
     year={1966},
     publisher={Holt, Rinehart and Winston}
   }

**Double Machine Learning**

.. code-block:: bibtex

   @article{chernozhukov2018,
     title={Double/debiased machine learning for treatment and structural parameters},
     author={Chernozhukov, Victor and Chetverikov, Denis and Demirer, Mert and 
             Duflo, Esther and Hansen, Christian and Newey, Whitney and Robins, James},
     journal={The Econometrics Journal},
     volume={21},
     number={1},
     pages={C1--C68},
     year={2018}
   }

**Mediation Analysis**

.. code-block:: bibtex

   @article{baron1986,
     title={The moderator--mediator variable distinction in social psychological 
            research: Conceptual, strategic, and statistical considerations},
     author={Baron, Reuben M and Kenny, David A},
     journal={Journal of Personality and Social Psychology},
     volume={51},
     number={6},
     pages={1173},
     year={1986}
   }

   @article{imai2010,
     title={A general approach to causal mediation analysis},
     author={Imai, Kosuke and Keele, Luke and Tingley, Dustin},
     journal={Psychological Methods},
     volume={15},
     number={4},
     pages={309},
     year={2010}
   }

**AI Evaluation Methods**

.. code-block:: bibtex

   @article{ribeiro2016,
     title={Why should I trust you?: Explaining the predictions of any classifier},
     author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
     journal={Proceedings of the 22nd ACM SIGKDD International Conference on 
              Knowledge Discovery and Data Mining},
     pages={1135--1144},
     year={2016}
   }

Example Citations in Text
-------------------------

When citing HAAM in your manuscript, you might write:

   *"We used the Human-AI Accuracy Model (HAAM; Li & Biesanz, 2025) to decompose 
   perceptual accuracy into direct and mediated pathways. The Percentage 
   of Mediated Accuracy (PoMA) was calculated using the Double Machine 
   Learning Lens Model Equation (DML-LME) as implemented in the HAAM 
   Python package (version 1.0; Li & Biesanz, 2025)."*

Or in the methods section:

   *"Following Li & Biesanz (2025), we applied the DML-LME framework to obtain 
   debiased estimates of mediation effects in our high-dimensional 
   setting. The analysis used cross-fitting with post-lasso regression for 
   feature selection and debiased coefficient estimation."*

Acknowledgments
---------------

HAAM incorporates ideas and methods from many researchers in psychology, 
statistics, and machine learning. We are grateful for the foundational 
work that made this integration possible.

If you have questions about citations or want to report usage of HAAM 
in your research, please contact us at raymond.li@psych.ubc.ca.