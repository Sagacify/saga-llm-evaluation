# Changelog

<!--next-version-placeholder-->

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
