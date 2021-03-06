# QEP Paper

## Requirements

Install [git lfs](https://git-lfs.github.com) to allow Git to work
with large files.

## Manuscript

- [Manuscript][Manuscript] (Google Doc file)
- [Figures](figures.key) (Keynote file)
- [TODO][TODO] (Google Doc file)

[TODO]: https://docs.google.com/document/d/1mDhirfAocMKSnjbefFEM5OWU6DWhoQlARX_NrqDh2Cg/edit
[Manuscript]: https://docs.google.com/document/d/1HA6aKhNrYh5xW34g0gtqVkfE5v0E02rYISAQpojsRAM/edit

## Experiments

- [Alternative splicing analysis][Alternative splicing analysis] (Google Doc file)
- [Source-code](https://github.com/glimix/horta-exp) (GitHub Repo)

[Alternative splicing analysis]: https://docs.google.com/document/d/19DvvZVtyyE1RO4Al_OsK83NrMqiU9WvE8MjBOrO65Ac/edit

## Technical report

Clone the repository via
```bash
git clone https://github.com/glimix/fastglmm-paper.git
```

Enter into the `tr` folder and compile the pdf:
```bash
cd qep-paper/tr
xelatex main.tex
open main.pdf
```

As you can see, `tr/main.tex` is the main file for compiling the whole
technical report.

## Datasets

- [Alternative splicing][Alternative splicing] (private link)

[Alternative splicing]: https://github.com/glimix/alternative-splicing
