# User study harness

`study/` holds the planned harness for a user study of the overlay (per `CLAUDE.md`'s repo-layout
notes). As of this writing:

- `study/logging.py` — empty stub, no content.
- `study/viewfinder.py` — empty stub, no content.
- `study/clips/` — directory present, contents not inventoried here.

**There is currently no design intent recorded anywhere in the repo for what these modules should
do** — no docstring, comment, or plan document specifies the study's methodology, what
`logging.py` should log, or what `viewfinder.py` should render. This doc is a placeholder noting
that gap rather than an invented design.

If you're picking this up: check with the user for the intended study design before writing
code here, or look for design notes outside this repo (e.g. the `paper/` write-up) that might
specify what data the study needs to produce.
