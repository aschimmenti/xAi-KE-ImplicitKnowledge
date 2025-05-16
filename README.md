# Information Extraction from Implicit Text: Evaluating and Adapting Large Language Models
Text Implicitness has always been a challenge in the field of Natural Language Processing (NLP), with traditional methods that rely on explicit statements to identify entities, relationships and events. For instance, "Maria attends church every Sunday" implicitly suggests that Maria's religion is Christian, where the sentence's meaning is inferred from a religious frame.  
Although this text does not require specific domain knowledge for comprehension, it presents a moderate level of complexity arising from relationships often inferred through contextual cues rather than explicitly stated.
Large language models (LLMs) have proven their effectiveness in NLP downstream tasks such as text comprehension, text generation, and information extraction (IE). 

This study examines how textual implicitness affects IE tasks in pre-trained LLMs: LLaMA 2.3, DeepSeekV1, and Phi1.5. 
We generate two synthetic datasets with Gpt-4o mini containing 10k implicit and explicit verbalization of biographies. 
The dataset is then used to analyze the impact of implicit and explicit information on LLM performance and investigate whether fine-tuning on implicit data improves their ability to generalize in implicit reasoning tasks.  

This research presents an experiment on the internal reasoning processes of LLMs in IE, particularly in dealing with implicit and explicit contexts. Results demonstrate that fine-tuning LLM models with LoRA (Low Rank Adaptation) improves their performance in extracting information over implicit texts, contributing to better model interpretability and reliability.

## License
The MIT License (MIT)

Copyright (c) 2015 Chris Kibble

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.