# Exact-byte metadata continuation

Prepared locally; **not dispatched**. V2 job **979604** failed before creating
outputs because text-mode reading normalized the retained series CSV's CRLF
line endings. Original metadata bytes were correct.

CPU **979619** ran the complete corrected inventory successfully: **522 complete,
3365 pending, one retained source buffer and one orphan raw cache**. It verified
the original CSV's exact bytes, both v2 roots absent and no original writes or
downloads. Receipt SHA:
`6c7cb68e72d0b3822ebc6894c38815e79988a256613770a5b1a5c19b617258ad`.

The only reconciliation change is comparing `read_bytes()` with the exact
original string's UTF-8 bytes. Corrected reconciliation SHA:
`1cb528bd3a716adcd74cba19305297103a1287b80dccbdb5ea2ee788f666ee4a`.
Actual receipt, original renderer regression and both failed predecessor
receipts remain required. Both predecessor jobs must remain FAILED 1:0 and
absent from the account-wide live queue; their output roots must remain absent.

Add `--prefix-admission-receipt` and `--prefix-admission-sha256` to the v2
launch arguments. Distinct output roots and ignored journal end in `-v3`.
All source numerics, retained-input behavior, mixed renderer provenance,
resources, time limits and funded reserves remain as reviewed for v2.
V1/v2 attempts and original data remain untouched.
