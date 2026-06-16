# A Mechanistic Understanding of Pronoun Fidelity in LLMs
**Abstract:** Faithful and robust pronoun use is important for fair and coherent generations, yet large language models largely fail when multiple referents use different pronouns.
To study the interplay of reasoning, repetition, and bias in this task, we provide a mechanistic, model-internal perspective, testing whether three mechanisms---group entity binding (G), recency bias (R), and stereotypical bias (S)---are causally implemented across several SOTA language models.
Using Boundless Distributed Alignment Search, we find all three coexist as causal subspaces distributed across network depth.
No single mechanism fully explains model behaviour, but a combination of the three consistently accounts for 91-99.5\%.
An attention head analysis further reveals two competing copying routes; group binding and stereotype share a localized concept-level route that retrieves a bound occupation-pronoun unit, while recency uses a distributed token-level route that repeats surface forms.
In sum, pronoun fidelity arises from competition between simultaneously active causal subspaces.
<p align="center">
<img width="350" height="450" alt="teaser" src="https://github.com/user-attachments/assets/04503e75-d83d-41be-8683-df3c04b498a9" />
</p>

