# Blog

## Run locally

1. Install node and yarn

2. Run `yarn install` then `yarn start`

## TODO

- [x] Code highlighting
- [x] Migrate to https://github.com/11ty/eleventy-base-blog
- [x] Repair TODOs in text.
- [ ] Table of contents (similar to https://staadecker.github.io/undergrad-assignments-post/)
- [ ] Show footnote on hover, maybe using littlefoot. (but littlefoot with hover doesn't work on mobile at all)
- ~~[ ] Setup heading anchors~~ (table of contents is enough) 
- [ ] Run Nutshell at build time rather than on the browser (but note that due to recursion this is not fully possible).
- [ ] Fix bug with Nutshells where when another heading starts with the same text (e.g. bootstrap) it is chosen instead

## Acknowledgments

- [littefoot.js](https://littlefoot.js.org/) for pretty footnotes.
- [KaTeX](https://katex.org/) for math rendering.
- [Nutshell](https://github.com/ncase/nutshell) by Nicky Case for expandable explanations.