# perseu

Winnin Project with Social Media.

In today's digital age, social media platforms like YouTube, Instagram, Facebook, Twitch, and TikTok serve as gold mines for unlocking invaluable cultural insights crucial for crafting effective marketing campaigns. These platforms offer a rich variety of user-generated content, providing a direct window into the behaviors, preferences, and trends shaping consumer culture. Here at Winnin, we use the data extracted from these platforms open APIs to find valuable insights to our clients.

You will be given a sample of posts data in CSV format. A Post is a video published in any of the platforms mentioned above. Be aware that different platforms have different metrics available through their APIs. Metrics which are not available for some platform are registered as 0. Different platforms may have different terminology regarding their content creators (e.g.: on youtube posts are published by channels; on instagram posts are posted by users/profiles etc). For simplicity, we will be calling every one of these entities Creator.

Based on the provided data, answer the following questions showing your work and/or any found evidence supporting your responses.

Which creators have had grown the most regarding their engagement? If engagement is not available, chose another metric which may indicate audience interest in them. Create this analysis in an timeframe that feels suitable to the problem (e.g.: monthly for the last year; weekly over the last month; daily over the last weeks etc)
Based on your answer in 1, for these content creators, make a prediction of how they will be in the future.
Build a function that takes a set of posts from a specific creator and explain the reasons for the growth/decline behavior of this creator. Apply it to the top 10 creators from questions 1 and 2.
You may use any opensource lib, database, public API or anything available to you in order to solve the problem. Ensure that your code runs and that your work is reproducible. Maintain clear and readable code. If you wish, use markdown blocks to explain your reasoning and/or guide the evaluator through your solution.
