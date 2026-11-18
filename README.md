# gta6-twitch-ccu-forecast

Entry for make.fun's gta6-twitch-ccu-forecast-01 bounty: forecast Twitch concurrent
viewers hour by hour for the GTA VI release day from hour 1 onward, as five quantiles
per hour.

The model is k-nearest-launch shape transfer over the comparables each case ships with:
rank past launches by the shape of their last 48 pre-release hours (a scale-free log
signature) and by their absolute viewer level, take the four nearest, transfer their
release-day ratio curve as the median path on top of hour 1's print, and size the
quantile bands from those neighbors' residual spread on the log scale.

Python 3.13 standard library only. Deterministic: no randomness, no network, both runs
write the same bytes.
