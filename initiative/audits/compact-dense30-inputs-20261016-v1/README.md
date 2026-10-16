# Actual dense30 MRI input acceptance

Root's original oneCPU/zeroGPU933517 COMPLETED successfully on evc33. Frozen
source243eca52, actual receiptSHA2b3186baab40f8f8e535891dd3930be94ecf0efb59d1cfef5290acfc6cb97ff6.
32actual MRI studies retain all2345original window tensors bit-exact while exposing
4188dense windows. Full metadata census expands fitting162651->305156windows and
outer84333->158073, about1.88xcoverage; fallback357/222counts are unchanged.
Every original center/window survives; no pixel payload copied and no labels,
source geometry, spatial field, contrast, model or loss changes. New centers must
have both physical source-neighbor edges, cap30per series/180per study.

This is actual MRI input/coverage evidence only. No encoder, GPU, optimizer or
accuracy ran. Target-device precision/throughput and fixed-checkpoint comparative
held-out predictions remain required before a competition candidate. More
windows may dilute attention or expose more noise, so coverage is a hypothesis,
not a proven benefit or a nine-hour guarantee. Devin is unavailable; root built
and admitted this directly. Issue96 closed as implemented, not as accuracy success.
