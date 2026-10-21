# Validation · 21 October 2026

## Executed model evaluation

Real OpenCV DNN YOLOX-S inference processed all 120 original PNG frames from the fixed 0011/0013 windows on CPU. No failed or poorly performing frames were removed. The same detections fed both trackers in chronological order, resetting tracker state between clips. One detector warm-up was excluded from timing. No labels enter the tracker, and forecast scoring consults future labels only after the causal pass.

There are 327 pedestrian box annotations (201 in 0011, 126 in 0013) and 12 sequence-scoped ground-truth track IDs. Motion tracker: 170 matched, 48 unmatched, 157 missed, 20 ignored predictions; 1 identity switch. Last-position tracker: 119 matched, 52 unmatched, 208 missed, 56 ignored; 8 switches. All confirmed reported tracks—including coasting predictions—are scored.

The 5-frame forecast sample has 131 eligible anchors and mean motion/stationary endpoint errors of 26.5595 / 70.2540 pixels. The 10-frame sample has 101 anchors and errors of 65.6208 / 133.5622 pixels. Pixel distances include ego-motion and are not physical distances. Overlapping horizons are correlated, and eligibility favors tracks that were already successfully detected and matched.

## Automated tests

`python -m unittest discover -s tests -v`: **41 passed, zero skipped**, Windows / Python 3.12 / OpenCV 4.13 / NumPy 2.3.5. The test suite covers detector decoding, confidence and class handling, model-hash rejection, matching/ignore precedence, optimal assignment against exhaustive enumeration, confirmation, gap coasting, expiration, ID ordering, deterministic reset, constant-velocity forecasts, covariance positivity, frozen snapshots, future-label scoring, recorded-metric reconstruction, HTTP serving and private/traversal-path rejection.

The integration study uses the real model. Unit tests are deterministic and require neither model nor dataset downloads. Committed report metrics are reconstructed by tests. JavaScript syntax is checked with Node. GitHub Actions completed successfully on Python 3.11 and 3.12 for release commit `f050c9fb6c4a91018c82773222ce0e324d43657d`: [verified test run](https://github.com/Ehsan-Ch/pedestrian-motion-lab/actions/runs/39567926906).

## Browser validation

[The public GitHub Pages preview](https://ehsan-ch.github.io/pedestrian-motion-lab/) was opened in the browser after the [successful deployment](https://github.com/Ehsan-Ch/pedestrian-motion-lab/actions/runs/39567942512). The hosted report loaded its metrics, sequence images and track overlays, and recorded playback was exercised. It requires no local server or model download.

The local report displays measured metrics and actual KITTI sequence images. Clip switching, recorded playback through frame 59 with automatic stop, 5/10-frame forecast selection, last-position tracker selection and ground-truth display were exercised in the browser. Both observed and coasting states are explicitly distinguished. Static replay does not perform new model inference.

Track selection, clearing the selection, frame stepping and JSON download were also exercised. The downloaded `crosswalk-0011-16.json` retained frame 16, the motion-tracker snapshot and eligible 5/10-frame forecast scores. The README preview is an actual browser screenshot. The report rendered without horizontal clipping in the observed 740-pixel-wide browser panel.

## Scope

This is a research prototype using a pretrained detector and an original motion tracker. It is not a trained autonomous-driving system, a reproduction of official SORT results, an official KITTI submission, or a safety validation. No live camera, physical vehicle, world-coordinate trajectory, personal identity, facial recognition or demographic inference was used. See README and ATTRIBUTION for the full protocol and licensing.
