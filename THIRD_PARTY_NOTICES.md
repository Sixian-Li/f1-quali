# Sources and third-party notices

Original code and documentation in this repository are licensed under MIT,
copyright 2026 Sixian Li. This does not relicense external datasets or source
documents. No source data archive, official HTML/PDF collection, original course
repository, or third-party library implementation is bundled.

## F1DB

Historical racing facts used in the fixed reconstruction come from
[F1DB](https://github.com/f1db/f1db), release `v2026.14.0`.
F1DB identifies its data license as
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Credit belongs to F1DB and its contributors. See the upstream repository for the
source license and its own acknowledgments.

This project normalizes identities and session tables, records explicit corrections
and exclusions, constructs qualifying labels and past-only features, fits models,
and reports aggregate evaluation statistics. These are project transformations,
not original F1DB outputs or an endorsement by F1DB. Source versions and checksums
are recorded in the packaged resources and benchmark provenance. Retain attribution
and describe changes when redistributing source data or derived data.

## f1schedule

Session schedule metadata is downloaded from
[theOehrly/f1schedule](https://github.com/theOehrly/f1schedule), pinned to commit
`5c807120ae154922fd0075c3583bc2bbf78b80ff`. Credit belongs to the project and its
contributors. Its README permits use, but this release does not assign it an MIT
or other standard license. The cached schedule files are not redistributed here.
Scheduled times remain distinct from actual session times and from publication times.

## FIA and Formula 1 materials

Official classification, timing, schedule and rule materials are referenced by
URL and checksum in the source manifests. Copyright in those materials remains
with their respective owners. Project review records preserve factual decisions,
source references and uncertainty; they do not distribute the original HTML/PDF
collection. Downloading or separately redistributing those materials is subject
to the source's terms. This project is independent and has no official affiliation.

## Software and course-method reference

NumPy, pandas, SciPy, PyArrow, Requests and development tools are installed as
dependencies, not vendored. Their own license notices remain applicable; exact
versions are recorded in `requirements.lock`.

The course-style baseline reconstructs the approach used in the author's earlier
STATS 507 project, with explicit prediction cutoffs and same-season rolling fits.
It is not a redistributed copy of the original course project. Experimental FastF1
downloads used during research are not necessary for this release.
