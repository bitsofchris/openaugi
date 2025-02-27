# Please someone - help me keep up with all my notes!

### Problem
I write, take voice notes, and I can't keep up.

How many times did I repeat this idea in the past?
Where is that other note that kind of spoke about this idea before?
How can I process my notes inbox?
What is the most important/impactful thing to work on next?

I have voice notes in Otter and thoughts/ ideas/ plans/ next steps in many notes in my Obsidian inbox (markdown files).

In data engineering, this is my raw data.

I want to use LLMs to process this - extract atomic ideas, tasks, know when it's a personal story that should be saved verbatim.

TLDR: I want my brain to keep moving as fast as it wants, but I don't want to stop and organize, structure, link, check if I've repeated myself in the past. I want an LLM to do this for me.

### Solution

##### 1. Distill Voice Notes/ Inbox notes - ETL on my notes
Start simple.

Given a brain dump or a single note - let's parse what it is.

Similiar to how Otter.ai works today, maybe even use their output for now.

This first distillation part I think can be done with just a LLM and some prompt engineering. And why not - LLMs have learned patterns in how humans think.

##### 2. Split, embed, de-duplicate
Given the output of our "processed" brain dump, let's take atomic items from it.

One idea, one task, one thought, one story (this is tricky b/c a personal story is a memory and we'd like to keep it together, perhaps a human can help us out and just label it for now so we use tagging to tell how to process it).

We then embed these atomic units from our procesed note.

Further research: embedding comparison vs just prompting an LLM to distill or de-duplicate ideas.

##### 3. Merge to existing vault (via PR)
Now this part I haven't really seen.

Given some new local data that we just processed above, let's compare it to our existing PKM/Knowledge vault/ etc.

Are any of these new ideas very semantically similar in the embedding space? (is this a duplicate thought, a new nuance to existing thought)
- track the metadata, count if it is we should use the frequency of thought as information

Where do the new atomic notes go? Do we link to existing projects? Existing ideas? Do we move tasks around?

Display this as a suggestion to the user - a pull request to your second brain.


# Journal

### 2025-02-27
Cutting edge research comes out everyday.

The secret sauce is not in how you distll, summarize, or what model you use.

It's in the community, the UX, the data -> the value you provide for people.

Whatever tech is used to do the distillation and retrival will improve over time, this part needs to be modular and model agnostic.

The value of what I want to build I think is the end-to-end experience, the conection of disparate tools and, the deep understanding of the use case because these are problems we all share here.

For this prototype the "distill engine" and the "graph" will just be done recursively using a LLM. Eventually we can explore more cutting edge techniques as they improve.

But I want to see end-to-end this working.

##### MVPs
MVP v0

(A) Voice Capture -> (B) Distill into Atomic Ideas -> (C) Connect together into Graph/ separate tasks from ideas

MVP v1

All the above -> (D) Connect to persistent/ existing knowledge repository (PR style UI)

##### Next Steps
Today I'm working on the v0 B -> C in a notebook using existing transcribed voice notes.

Once that is working I'll make a UI around it where I can paste in the voice note and see the newly created atomic notes and their visual connections.

Then from their add voice transcription - 1 button, Whisper API?, and pass the transcription to the above.

At this point I think we have a cool demo to illustrate this idea of Voice Capture -> Self-Organizing Second Brain


### 2025-02-26

Trying to keep this scope small. But the problem is I have a lot of big brain dump voice notes. 

1. Distill them into atomic ideas. 

This is the first step.

I want to explore https://github.com/OpenSPG/KAG and the https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/hierarchical/ to help with this.

I also don't want to make things too complicated. Maybe it's just:

- full text -> summary
- sections + full text summary -> section summaries
- 2-3 sentences + section summary -> chunk summaries
- distill to unique, atomic ideas - look at frequency of ideas and similar concepts

all this might be done with a LLM? no need to embedding similarity search yet?

Then you maybe classify the atomic idea -> note, task, idea, question (as metadata?)

Then do the embeddings and compare to existing vault?