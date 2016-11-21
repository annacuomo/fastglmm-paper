# Readme

## Technical report

We use TM-style-class for modern LaTeX document class: https://github.com/qTipTip/TM-style-class
Please, follow the Installation section because you __NEED__
[Lato](http://www.latofonts.com/lato-free-fonts/) and
[Source Code Pro fonts](https://github.com/adobe-fonts/source-code-pro).

As a quick start, just download the TTF files for Lato fonts and double-click
on every font file.
They get installed in that way.

For the Source Code Pro fonts, you can proceed as follows:
```bash
brew tap caskroom/fonts
brew cask install font-source-code-pro
```

You can now enter into the `tr` folder and compile the pdf:
```bash
cd tr
xelatex main.tex
open main.pdf
```
