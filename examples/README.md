# Offline example

Run `f1-quali demo --output demo-run` after installation. The generator in
`f1_quali.demo.synthetic_dataset` creates 20 entirely fictional drivers, 20 historical
events in 2023 and six target events in 2024. It uses seed `20260926`.

The command writes a checksummed dataset, prior-event rating snapshots, features,
an annual m1r1 model (four-pool heads plus the shared rating corrections), predictions
and event metrics. Compare `demo-run/summary.json` with
[expected_demo_summary.json](expected_demo_summary.json). `f1-quali demo --method v6`
runs the previous release method; compare it with
[expected_demo_summary_v6.json](expected_demo_summary_v6.json).

There are 120 prediction rows across six target events. No real driver results or
pretrained model are bundled. The example is a functional test, not a performance claim.
