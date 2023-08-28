# DeepAtash

## General Information ##
This repository contains the source code and the data of the paper "DeepAtash: Focused Test Generation for Deep Learning systems" by T. Zohdinasab, V. Riccio, and P. Tonella. published in the Proceedings of the ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2023).

## Getting Started ##

Click [here](./documentation/getting_started.md) to see how to set up DeepAtash and validate its general functionality by using our pre-configured Docker image.

## Detailed Description ##

Click [here](./documentation/details.md) to see how to validate the paper’s claims and results in detail. 

## Repository Structure ##
The package is structured as follows:

* [__MNIST__](./MNIST) contains the DeepAtash tool adapted to the handwritten digit classification case study and the instructions on how to use it;
* [__IMDB__](./IMDB) contains the DeepAtash tool adapted to the movie sentiment analysis case study and the instructions on how to use it;
*  [__BNG__](./BNG) contains the DeepAtash tool adapted to the self-driving car case study and the instructions on how to use it;
* [__experiments__](./experiments) contains the raw experimental data and the scripts to obtain the results reported in the ISSTA paper;
* [__documentation__](./documentation) contains a quick installation guide and a detailed description of the tool.


_Note:_ each sub-package contains further specific instructions.

## Reference

If you use our work in your research, or it helps it, or if you simply like it, please cite DeepHyperion in your publications. 
Here is an example BibTeX entry:

```
@inproceedings{tahereh2023deepatash,
  title={DeepAtash: Focused Test Generation for Deep Learning Systems},
  author={Tahereh, Zohdinasab and Riccio, Vincenzo and Paolo, Tonella and others},
  booktitle={Proceedings of the ACM SIGSOFT International Symposium on Software Testing and Analysis},
  year={2023}
}
```

## License ##
The software we developed is distributed under MIT license. See the [license](./LICENSE.md) file.

## Contacts

For any related question, please contact its authors: 
* Tahereh Zohdinasab ([tahereh.zohdinasab@usi.ch](mailto:tahereh.zohdinasab@usi.ch)) 
* Vincenzo Riccio ([vincenzo.riccio@usi.ch](mailto:vincenzo.riccio@usi.ch))
* Paolo Tonella ([paolo.tonella@usi.ch](mailto:paolo.tonella@usi.ch)).

## DOI