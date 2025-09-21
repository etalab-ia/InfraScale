# InfraScale

Infrascale is a tool developed by the Albert API team to estimate GPU requirements to do LLM inference at scale. It makes estimations within a 10% error margin for models between 8b and 27b and for node sizes lower than 8.

## Why InfraScale?

While working within the French public administration, we realized that nearly every organization faces the same question: how many GPUs are needed to reliably serve a given number of users?
Operating the official GenAI API for the government, we confronted this challenge firsthand. To plan for the near future—scaling to 2.5 million public agents—we needed a way to benchmark and compare different approaches (e.g., commercial services versus self-hosted open-weight models). InfraScale was born from this necessity.

## Quickstart

- Create and activate a python virtual environment
- Install required dependencies with `pip install -r requirements.txt`
- Run with `streamlit run app.py`

## How it works

We want to find the smallest number of GPUs that manages to do inference with regards to the targets (number of concurrent users, type of GPU, inference model, target throughput, target wait time, etc.) given by the user.

We define the following variables :

- Let $S \in \mathbb N_+^*$ the number of GPUs in a given node, i.e the number of GPUs that a single copy of the model is loaded onto, corresponding to [model parallelism](https://colossalai.org/docs/concepts/paradigms_of_parallelism/#model-parallel).
- Let $N \in \mathbb N_+^*$ the number of nodes, i.e the number of parallel copies of the model that are loaded, corresponding to [data parallelism](https://colossalai.org/docs/concepts/paradigms_of_parallelism/#data-parallel).
- Let $B \in \mathbb N_+$ such that the max request batch size is equal to $2^B$.
- Let $Q \in \mathbb R_+$ be the expected ratio of concurrent requests to batch size for any node at any given time. $Q \leq 1$ means that requests will be processed immediately on average. $Q > 1$ means that requests will wait for previous requests to be processed, this is possible without having a queue expending dramatically as long as, on average, $Q \times batch\textunderscore size$ requests can be processed quicker that the time it takes for the same amount of requests to be sent by users, which will be implied by our throughput constraint.

### Constraints

Given the above variables, we lay out four constraints :

#### Memory

$total\textunderscore mem\textunderscore required \leq S' \times vram\textunderscore one\textunderscore gpu$

With $S' = 1 + (S-1)\times (1 - parallelism\textunderscore mem\textunderscore efficiency)$. We consider $parallelism\textunderscore mem\textunderscore efficiency = 0.85$. A given node should have enough available VRAM to load model weights and do inference (which means having enough spare memory for activations and KV cache).

#### Throughput

$target\textunderscore throughput \leq throughput\textunderscore one\textunderscore gpu \times f(S) \times g(B, Q) \times \frac{1}{Q}$

The throughput in tokens per second for any given request should be higher than the target set by the user. Currently, we use a theoretical model (see `constraints.py` and `calibration.ipynb`) of the throughput for one request on a single GPU and appromixate its relation with node size $S$ and GPU load (which is a factor of batch size and $Q$). We take wait time into consideration here, thus our model accounts for prefill time and average queue size $Q$ (the unit of Q is number of batches).

#### Capacity

$concurrent\textunderscore users \times reqs\textunderscore per\textunderscore user \leq N \times Q \times 2^B$

The whole system should be able to serve the expected number of concurrent users. We currently consider $reqs\textunderscore per\textunderscore user=1$ which is unrealistic, the number is expected to be 10 times lower but has yet to be estimated empirically based on Albert API's usage metrics.

#### Wait

$target\textunderscore wait\textunderscore time \geq TTFT + \max (0,Q-1) \times time\textunderscore one\textunderscore batch$

Wait time before the beginning of an answer should be lower than the target set by the user. The terms $\max (0,Q-1) \times time\textunderscore one\textunderscore batch$ express the fact that if $Q \leq 1$, the absence of a queue means that there is no wait for previous requests. Note that the TTFT sets a lower bound for target wait time.

### Optimization problem

We solve the following constrained optimization problem with [Nelder-Mead's algorithm](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) :

$$
\underset{S,N,B,Q}\min  S \times N {} s.t \begin{cases}
   total\textunderscore mem\textunderscore required \leq S \times vram\textunderscore one\textunderscore gpu &\text{(memory)} \\
   target\textunderscore throughput \leq throughput\textunderscore one\textunderscore gpu \times \frac{1}{Q} \times f(S) \times g(B) &\text{(throughput)} \\
   concurrent\textunderscore users \times reqs\textunderscore per\textunderscore user \leq N \times Q \times 2^B &\text{(capacity)} \\
   target\textunderscore wait\textunderscore time \geq TTFT + \max(0, Q-1) \times time\textunderscore one\textunderscore batch &\text{(wait)} \\
\end{cases}
$$

We don't look for an analytical solution as we except to make updates to our theoretical model and to this optimization problem in the near future.

## Limitations

Our framework has been tested on models from 1b to 27b in the LLama, Mistral, Qwen and Gemma families, on 1 H100 GPU and on 2 tensor-parallel H100 GPUs with PCIe intercomms (see `calibration.ipynb`, although it does not cover every test that has been made). Some tests have been made on 1 A100 GPU aswell for mistral-small-24b and llama-3.1-8b. We implicitely make the following hypothesises:

- Requests are questions with a 20/80 prefill to decode tokens ratio. Tests haven't been made with summary or coding requests that have a much higher prefill to decode ratio.
- GPUs of a same node are linked with SXM intercomms.
- The default parameters of VLLM (or suggested parameters for mistral and qwen models) are used.

## Next steps

- Make empirical tests on bigger node sizes (4 and 8), bigger models, varying intercomms bandwidth and different request types.
- Find a more robust approximation of the relation between throughput, $S$, $B$ and $Q$ (related functions are `calculate_scaling_with_S` and `calculate_scaling_with_B_and_Q` inside `constraints.py`).
- Estimate $reqs\textunderscore per\textunderscore user$.
- Take GPU intercomms into account for throughput and wait constraints.

## Contributing

The list of available GPUs and models in Infrascale can be found in `db/gpu.json` and `db/models.json`. New GPUs/models can be added by appending them directly at the bottom of the json files and following the current data format.

Locales are set in the `translations` folder. When adding a new language, don't forget to update the LANGUAGES dictionnary in `language.py`.

The underlying logic is found in `solver.py` and `constraints.py`. Notebooks `solver.ipynb` and `calibration/calibration.ipynb` may be useful to understand the logic and how it was defined. A comprehensive article is currently being written to further explain how Infrascale works. 

## Authors and reference

Thanks to the authors Jules Pondard, Théo Lartigau, Cyril Lay and Rémy Siahaan--Gensollen.
