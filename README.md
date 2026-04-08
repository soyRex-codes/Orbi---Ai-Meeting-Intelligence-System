# Orbi-AI Meeting Intelligence System
## What this is
An AI system that processes meeting recordings and extracts actionable intelligence like who said what, what was decided, and what needs to happen next.

## Why we are building this
We had a 2-hour meeting with many discussions, but we couldn’t keep track of all the decisions, tasks, who said what, or important dates. It was frustrating, and we had to go back and watch the Zoom recording, which was honestly very boring.

So, we came up with the idea of automating this process so others don’t face the same problem.

## Current Status
Project setup complete. Pipeline not yet functional.

## What we are learning
We have documented our learning journey in docs/learning_journal.md

# Testing pipeline for audio_processor and whsiper_transcriber

- testing both together revealed that they both work correctly as expected togther, audio gets converted to 16khz, to mono and other error checks and then gets passed to trasnscriber to transcribe, processing time. 
- checks were performed with different models such as tiny, base medium and large, and was found out that each larger model performs little better than previous smaller model in recognizing words, speech and full context, while it was also found out that in some cases smaller model were better able to identify what speaker was speaking which was very surprising, so it was concluding that each models have there uniqe strengths even if the sizes differs.
- when users have thicker accents eg: phlipino, or affircan, the model struggles to identify their speach but works great when listens modern english native speakers, which points that mdoel maybe trained highly on english native speakers first.


