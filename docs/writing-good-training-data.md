# Writing good training data

The quality of your model is mostly the quality of your Q&A pairs. This guide covers
what to write, how much of it you need, and what fine-tuning can and can't do for you.

## The format

Training data is plain text, in `Q:` / `A:` pairs:

```
Q: What is the capital of Australia?
A: Canberra. Many people guess Sydney, but Canberra has been the capital since 1913.

Q: Who are you?
A: I'm TestAI, a friendly assistant built with LLM Training Studio.
```

Add pairs in the Studio's **Training Data** tab (they are saved to
`models/<ModelName>/knowledge/manual/manual_entries.txt`), or drop your own `.txt`
files anywhere under `models/<ModelName>/knowledge/`. Answers can be several lines long.

## What fine-tuning is good at

- **Voice and personality.** Tone, formality, humour, how long answers are, how it
  introduces itself. This is where fine-tuning shines, and it needs surprisingly little data.
- **Format.** Always answering in a certain structure, always giving a short answer first.
- **A focused set of facts.** Your product, your game world, your family trivia, your
  team's procedures. Things the base model could not possibly know.

## What it is not good at

- **A large, changing document collection.** If you want answers from hundreds of pages
  that change often, a search-based approach (retrieval) is a better fit. Fine-tuning
  bakes knowledge in at training time; it does not look anything up.
- **Guaranteed accuracy.** A fine-tuned model can still make things up, especially for
  questions close to, but not covered by, your data. Test it with questions you did not train on.

## How much data?

Rules of thumb, not laws:

| Goal | Pairs to start with |
|---|---|
| Give the model a name and personality | 20-50 |
| Teach a small topic area | 50-200 |
| Cover a broad subject well | several hundred or more |

The bundled **TestAI** sample builds 45 examples, which is enough to see the whole
workflow work. More *varied* pairs beat more *repeated* pairs.

## Habits that make a big difference

1. **Ask each important fact in several ways.** People will not phrase a question the way
   you wrote it. Write 3-5 versions ("What's the capital of Australia?", "Which city is
   Australia's capital?", "Where does Australia's government sit?") with consistent answers.
2. **Write the answers the way you want the model to speak.** It copies the style of your
   answers, so if they are terse, it will be terse.
3. **One fact per answer, stated plainly.** Long answers that bundle many facts are
   harder for the model to learn.
4. **Keep answers consistent.** If two pairs contradict each other, the model will
   learn the contradiction. Build Dataset flags duplicates and empty pairs for you.
5. **Include a few "I don't know" examples** for things outside your topic, so the model
   learns to say so instead of guessing.
6. **Use your real system prompt.** Whatever you set in Model Settings is baked into the
   training examples and into the exported model.

## Check your result

After training and exporting, chat with the model and ask:

- questions straight from your data (it should get these right),
- the same questions reworded (this is the real test),
- a few things you never covered (does it admit it doesn't know?).

If it fails on reworded questions, add more phrasings. If it parrots answers word for word
on unrelated questions, you may be training too long: set fewer epochs.

**Why the Studio trains longer on small datasets:** a model only memorizes a fact after many
small learning steps. In our tests, 45 Q&A pairs trained for 18 steps reproduced none of the
answers; the same data trained for about 270 steps reproduced most of them, a third word for
word. So with the default Auto setting, the fewer pairs you have, the more passes it makes.
