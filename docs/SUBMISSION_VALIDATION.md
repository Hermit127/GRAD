# Validation performed for this package

- Parsed all packaged Python source files without executing experiments.
- Recomputed T3 aggregate from 240 recorded cases; ASR, BU and UA matched the
  six archived method/backbone cells.
- Reran W4 on 232 examples in 58 groups using the packaged nine final run outputs;
  the GRAD/Text/Temporal mean AUROCs were 0.9699597041 / 0.8635552913 /
  0.9169474171. Endpoints remain separately defined in the output JSON.
- Seven canonicalization tests passed.
- Eleven closed-loop offline runner tests passed, including prefix visibility,
  blocked-action handling, task isolation, error accounting and gate behavior.
- All tensors in 12 anonymized W1/gate checkpoints were verified exactly equal
  to the corresponding original checkpoint tensors.

The lightweight runtime initially lacked PyTorch. Neural gate tests were then
run in the available Python 3.11/PyTorch 2.14 environment and passed. No new
training, upstream-baseline installation or paid API call was performed.
Full CAA retraining and all historical integrations were not rerun. See
DATA_AND_DEPENDENCIES.md for inputs that are outside this package.
