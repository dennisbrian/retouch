# Edit reports and Content Credentials

Two opt-in batch options record what retouch did to each photo:

- `--edit-report` writes a plain "what was changed" report per photo.
- `--sign-cert` / `--sign-key` sign each retouched photo with
  [Content Credentials](https://contentcredentials.org) (C2PA), using a
  certificate you supply. The same report is embedded inside the photo.

Both are command-line options (`cli.py` or `./run batch`). The app's
one-photo export and the GUI batch tab don't offer them yet.

## Edit report

```bash
./run batch shoot/ -o out/ --recipe cosplay_clear_v1 --edit-report
```

Each photo gets `out/edit-reports/<file name>.json` (one folder per output
folder, so a recursive batch keeps them beside their photos). It lists:

| Field | What it says |
|---|---|
| `recipe`, `faces_detected` | The recipe that ran and how many faces it found. |
| `shape_changed` | `true` when any face or body **shape** setting was active (face slimming, jaw/nose/eye reshaping, body reshaping). Several countries and platforms require reshaped images to be labelled; this is the flag to check. |
| `edits` | Every active setting, grouped: face and body shape, skin, makeup, eyes/teeth/glasses, hair/wig/costume, relighting, background, AI models, colour/tone/finish. Values are what the engine actually ran with, recipe values included. |
| `ai_used`, `generative_fill` | Model-backed steps (for example NAFNet noise reduction). retouch has no generative fill, so `generative_fill` is always `false`. |
| `pixels` | Share of the frame that changed by more than 3 of 255 levels, and the mean change, measured on a 1024 px copy. A cross-check that the listed edits did something. |
| `not_applied_no_face` | Face edits the recipe asked for that could not run because no face was found. |
| `source`, `output` | File names and SHA-256 hashes of the original and the retouched file. |
| `content_credentials` | Whether the source had Content Credentials and whether the output was signed. |
| `summary` | The same information as short sentences, for pasting into a client note. |

A setting being listed means it was switched on, not that its effect is
visible: for example, face slimming is faded out on faces turned far to the
side. `pixels` shows how much changed overall.

## Camera Content Credentials are no longer copied

Some cameras sign their photos. That signature covers the exact pixels the
camera captured, so it can't be valid on a retouched photo. Earlier versions
copied the camera's C2PA block into the retouched JPEG unchanged, and
verifiers then reported the file as **Invalid** (`assertion.dataHash.mismatch`),
which reads as "this photo was tampered with".

retouch now leaves the camera's block out of every export. The edit report
notes that the source had credentials. To carry them forward properly, sign
the output (below): the camera's credentials are then kept inside the new
ones as the original photo's history.

## Signing with your own certificate

Signing needs the optional `c2pa-python` package (Python 3.10 or newer):

```bash
uv sync --extra desktop --extra credentials   # or: uv pip install c2pa-python
```

Then pass your certificate chain and private key as PEM files:

```bash
./run batch shoot/ -o out/ --recipe natural \
    --sign-cert ~/keys/chain.pem --sign-key ~/keys/signing.key
```

| Option | Meaning |
|---|---|
| `--sign-cert PEM` | Your signing certificate followed by any intermediate certificates (not the root). |
| `--sign-key PEM` | The matching private key, unencrypted PKCS#8 (`-----BEGIN PRIVATE KEY-----`). It is read only to sign and is never logged or copied. Convert other key files with `openssl pkcs8 -topk8 -nocrypt -in old.key -out signing.key`. |
| `--sign-alg` | Key type: `es256` (default, an ECDSA P-256 key), `es384`, `es512`, `ps256`, `ps384`, `ps512` or `ed25519`. |
| `--sign-tsa URL` | Optional RFC 3161 timestamp server. A timestamp keeps the signature valid after the certificate expires. |

Each signed photo carries:

- a `c2pa.opened` action naming the original photo as its parent, and the
  camera's own credentials inside that parent when it had them;
- `c2pa.edited`, `c2pa.color_adjustments` and `c2pa.filtered` actions that
  describe the edits in the same groups as the report, with "Face or body
  shape changed" as its own action;
- the full edit report as the `org.retouch.edit_report` assertion.

If a photo can't be signed, its unsigned output is deleted and the photo is
reported as failed, so re-running the batch retries it rather than skipping
it. RAW and HEIC originals can't be read by the C2PA library, so for those
the decoded photo (as retouch saw it before editing) is recorded as the
parent instead, and the report says so. JPEG, PNG, WebP and TIFF outputs
can be signed; EXR can't.

### Which certificate

Verifiers such as [contentcredentials.org/verify](https://contentcredentials.org/verify)
show a signer as **trusted** only when the certificate comes from an issuer
on the C2PA trust list. Such certificates are sold by certificate
authorities; retouch neither bundles nor buys one.

Any other certificate still produces a signature that verifies as intact and
unaltered (`validation_state: Valid`), and any later change to the photo
makes it `Invalid`, but the signer shows as unknown
(`signingCredential.untrusted`). That is enough for a studio's own records
or for testing. A self-made one takes a few `openssl` commands:

```bash
# A private root for your studio (keep ca.key offline)
openssl ecparam -name prime256v1 -genkey -noout -out ca.key
openssl req -new -x509 -key ca.key -days 3650 -out ca.pem \
    -subj "/CN=My Studio Root/O=My Studio" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"

# The signing key and certificate retouch uses
openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out signing.key
openssl req -new -key signing.key -subj "/CN=My Name/O=My Studio" -out signing.csr
printf "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=emailProtection\n" > signing.ext
openssl x509 -req -in signing.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -days 825 -extfile signing.ext -out chain.pem
```

The C2PA library refuses a self-signed certificate, which is why the signing
certificate is issued from a separate root.

### Checking a signed photo

```bash
.venv/bin/python -c "from retouch.content_credentials import read_credentials as r; import sys; c=r(sys.argv[1]); print(c['validation_state'], c['codes']); print(*c['report']['summary'], sep='\n')" out/photo.jpg
```

or drop the file on contentcredentials.org/verify.

## Python API

```python
from retouch.edit_report import build_edit_report, write_edit_report
from retouch.content_credentials import SigningConfig, sign_output, read_credentials

result = engine.process(img, recipe="natural")
write_image_with_icc("out.jpg", result)
report = build_edit_report(result.params, source_path="in.jpg", output_path="out.jpg",
                           face_count=result.face_count, before=img, after=result)
write_edit_report(report, "out.jpg")

cfg = SigningConfig.from_paths("chain.pem", "signing.key")
embedded = build_edit_report(result.params, face_count=result.face_count,
                             source_path="in.jpg", output_path="out.jpg",
                             signed=True, include_output_hash=False)
sign_output("out.jpg", embedded, cfg, source_path="in.jpg", source_preview=img)
print(read_credentials("out.jpg")["validation_state"])
```
