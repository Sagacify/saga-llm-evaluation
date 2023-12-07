# Changelog

<!--next-version-placeholder-->

## v0.7.0 (2023-12-07)

### Feature

* **project:** Llama model can be now load from local, minor fix, improve doc ([`6064ffc`](https://github.com/Sagacify/saga-llm-evaluation/commit/6064ffc5e7d4d767f9316209a7efffdf39548e68))
* **project:** Redesigned scorer, llm_metrics support lists as input, scorer test skipped ([`99932ea`](https://github.com/Sagacify/saga-llm-evaluation/commit/99932eaf97d95b52eb7458b84f7948a8e1e660cc))
* **llm_metric:** For each llm_metric, evaluation model can be passed as input ([`d01ca44`](https://github.com/Sagacify/saga-llm-evaluation/commit/d01ca44498615b0c58a59823e603ad39512f256d))

### Fix

* **llm_metrics:** Unify default value for models ([`c467e28`](https://github.com/Sagacify/saga-llm-evaluation/commit/c467e28b58f8e3e2746d069d2bde2e85d961abe3))

### Documentation

* **language_metrics:** Adjust doc for q_squared compute function ([`8fa7e97`](https://github.com/Sagacify/saga-llm-evaluation/commit/8fa7e97ae9d0eff10be31ef9846a8e6100e5cad8))

## v0.6.0 (2023-10-26)

### Feature

* **add_aspects:** Add aspects to geval and gptscore small changes in arguments ([`d76676d`](https://github.com/Sagacify/saga-llm-evaluation/commit/d76676d7fb296e1306e879f01716c76515ef0125))
* **scorer:** Add scorer class support ([`c457d31`](https://github.com/Sagacify/saga-llm-evaluation/commit/c457d3131ddf4d61af1b94a8902b44763353e9db))

## v0.5.0 (2023-10-25)

### Feature

* **llm_metrics:** Refactor gptscore and add support for geval and selfcheck ([`980acdb`](https://github.com/Sagacify/saga-llm-evaluation/commit/980acdb0014d9b8c6291a5449f4ae61e3a6b9ee4))
* **gptscore:** Revamp to make multiple predictions at once ([`ce999f8`](https://github.com/Sagacify/saga-llm-evaluation/commit/ce999f88f6a0230a00547fd092f951700b93b044))
* **gptscore:** Add gptscore support ([`09d2cba`](https://github.com/Sagacify/saga-llm-evaluation/commit/09d2cbaf6354cb7d728fd7758bc7fba7c51c58fb))

### Fix

* **format:** Run pylint and black ([`7986b7f`](https://github.com/Sagacify/saga-llm-evaluation/commit/7986b7f580e93ef20cdb57d132144f880bb077aa))
* **pylint:** Format code ([`4f44004`](https://github.com/Sagacify/saga-llm-evaluation/commit/4f44004ea6b7a21a0ee2c86de1883e969942bc9a))

## v0.4.0 (2023-10-23)

### Feature

* **language_metrics:** Add q squared ([`329b7bd`](https://github.com/Sagacify/saga-llm-evaluation/commit/329b7bd4631500dc3a24538268ac79afb6ee2b1b))

### Fix

* **versions:** Update versions of tensorflow back ([`a25acd7`](https://github.com/Sagacify/saga-llm-evaluation/commit/a25acd7f6c70c99b35ee95512402cd4aebc4ec8f))

## v0.3.0 (2023-10-19)

### Feature

* **metadata:** Add support for computing simple metadata ([`1f12800`](https://github.com/Sagacify/saga-llm-evaluation/commit/1f128005fd755a8d3ff7eeaaebe6113b76ba721c))

### Fix

* **ci:** Downgraded elemeta to 1.0.7 ([`c8c872c`](https://github.com/Sagacify/saga-llm-evaluation/commit/c8c872cebf918062af03e7929988e6a086541229))
* **ci:** Attempt for fixing ci ([`74b2078`](https://github.com/Sagacify/saga-llm-evaluation/commit/74b2078d0bd306880271a2045d75215eb8e1973d))

## v0.2.0 (2023-10-18)

### Feature

* **bleurt:** Add bleurt metric support ([`5a315c6`](https://github.com/Sagacify/saga-llm-evaluation/commit/5a315c6d907ffc5590b8929d89df9fcc84579865))

### Fix

* **tensorflow-macos:** Attempt to fix poetry install for ci ([`9e1cfb6`](https://github.com/Sagacify/saga-llm-evaluation/commit/9e1cfb603b9746b459efc9db68049fcc99d1d645))

## v0.1.0 (2023-10-18)

### Feature

* **embedding_metrics:** Implement BERTScore and MAUVE and add unit tests support ([`04d50ff`](https://github.com/Sagacify/saga-llm-evaluation/commit/04d50ff33b9dd4acf9740b38a159208d5ccce94d))
* **project:** Add initial structure ([`786afbb`](https://github.com/Sagacify/saga-llm-evaluation/commit/786afbbf86995cdca8ec121e07b25c75ee8b60a3))

### Fix

* **cd:** Change version for python semantic release ([`a84c790`](https://github.com/Sagacify/saga-llm-evaluation/commit/a84c7906b3c6f1c3443e8637a710bd86cbe7c50b))
* **embedding_metrics:** Add featurize_model_name argument to init for mauve score ([`db39ad3`](https://github.com/Sagacify/saga-llm-evaluation/commit/db39ad31dd226b24956cb39786198900cf937719))
* **embedding_metrics:** Add tests, attempt to fix libcublas and move pytest to dev dependencies ([`1d45ec8`](https://github.com/Sagacify/saga-llm-evaluation/commit/1d45ec88ee9c97b80888cfe7e9561b4b51ae0f8d))
* **embedding_metrics:** Change default models for bertscore and mauve ([`8e9a20d`](https://github.com/Sagacify/saga-llm-evaluation/commit/8e9a20d1484cc597f8370ac4036e386c418f045a))
