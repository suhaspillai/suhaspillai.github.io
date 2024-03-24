---
layout: post
title: Question and Answering system using Large Language Model (llama-2)
---

This post involves building question and answering system to better understand the behaviour and prediction of stocks and crypto currency. The important components to building this system are the following

1. Open Source Large Language Models such as [Llama-2](https://ai.meta.com/llama/)
2. [Open AI (GPT turbo 3.5 or GPT-4.0)](https://openai.com/)
3. [Financial data from Yahoo Finance](https://pypi.org/project/yfinance/) and [finnhub](https://finnhub.io/)
4. [Modal](https://modal.com/) for training large language models
5. Transformer library from [HuggingFace](https://huggingface.co/docs/transformers/index)

The following diagram demonstrates end-to-end process.

<figure>
<p align="center">
  <img src="{{ site.baseurl }}/images/Q_and_A_whole_process.png">
</p>
<figcaption>
<b> 
<p style="color:black;font-size:15px" align="center"> Q & A process end to end </p>
</b>
</figcaption>
</figure>

There are 3 main parts to finetuning which are decribed below.

<H2>Data creation:</H2>
The data creation module consists of 3 parts

<H5>1. Yahoo Finance:</H5>
One of the most valuable resources for monitoring the daily fluctuations in stock prices is Yahoo Finance. This platform offers comprehensive data, including daily, weekly, and monthly insights into stock prices.  The yfinance API in Python serves as a powerful tool, providing useful information related to stock prices. I extract weekly information of the stock prices and categories the weekly movement in stocks as either going Up or Down along with %.

<H5>2. Finnhub:</H5>
Finnhub.io offers a Python-based API that facilitates the extraction of information pertinent to stocks / cryptocurrency. Leveraging Finnhub, I retrieve all news articles associated with a specified stock or cryptocurrency.

<H5>3. GPT-turbo 3.5 or GPT-4.0</H5>
The objective is to generate question and answering pairs for training the Llama-2 model. Using OpenAI's API, I supply information encompassing all the news associated with a specific stock, including its trend (either upward or downward). Then I ask GPT-4 to provide pros and cons about the stock based on the news information and provide analysis on why the stock might go up or down. Following is the code snippet to call open ai's API 

<pre>
  <code style="font-size: 16px;">
  completion = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                          ]
                    )
  #system prompt: Telling GPT what is the role.
  #user prompt: Information related to the stock / cryptocurrency
  </code>
</pre>


<H2>Finetunning:</H2>
Once the data is generated for question and answering its time to train llama-2. Meta provides different llama-2 models, I am using llama-2 7B model from huggingface. Rather than finetuning all the weights of llama-2, I use LoRA (Low-Rank Adaptation) technique to fine tune llama-2. LoRA significantly reduces the number of trainable parameters by inserting smaller number of new weights into the model and training on them. Its space and memory efficient. Huggingface provides all the required API's to train llama-2 using LoRA. Following is the LoRA config used for training

<pre>
  <code style="font-size: 16px;">
  peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      bias='none',
  )

  model = get_peft_model(model, peft_config)
  </code>
</pre>



<H2>Training on the cloud using Modal</H2>
Modal offers a convenient solution for executing code in the cloud. Notably, it provides a swift and hassle-free approach to training extensive language models, eliminating concerns about GPU memory limitations. All that is required is to specify the necessary libraries/packages for code execution. In essence, Modal initiates containers in the cloud, providing a seamless method for training large language models. Following are the basic steps

<pre>
  <code style="font-size: 16px;">
# step 1: Create a container image
llm_train_image=(modal.Image.debian_slim().pip_install(
        "datasets",
        "matplotlib",
        "scikit-learn",
        "torch",
        "accelerate==0.21.0",
        "peft==0.4.0",
        "bitsandbytes==0.40.2",
        "transformers==4.31.0",
        "trl==0.4.7"        
    )
    .run_function(download_models)
    .pip_install("wandb==0.15.0")
    .pip_install("rouge-score")
    .pip_install("scikit-learn")
    ) 

# step 2: Create a stub that uses the image
stub = modal.Stub("train_llm", image=llm_train_image)

# step 3: Add this decorator on top of the function, all the code inside can use the packages from the image
@stub.function(mounts=[modal.Mount.from_local_dir("/home/suhaspillai/Suhas/git/llms/ask-fsdl/fin_llm/src/",
 remote_path="/root/")], gpu="A100", timeout=30000, network_file_systems={model_store_path: vol})
def load_dataset():
  --- Training code ---
  </code>
</pre>


Once training is finished I plot training and validation loss for the trained model trained on A-100 GPUs for 100 epochs.

<figure>
<p align="center">
  <img src="{{ site.baseurl }}/images/Q_and_A_training_validation_loss.png">
</p>
<figcaption>
<b> 
<p style="color:black;font-size:15px" align="center"> Training and Validation loss </p>
</b>
</figcaption>
</figure>




Once the model is trained, we can provide it with the news articles for a specific week and it would provide pros, cons, prediction and analysis about the stocks and cryptocurrency. Following is the output from llama-2 and FT-llama-2. 


<!--  <div class="split-container"> -->
<div class="third-textbox" id="leftTextbox" contenteditable="true"> 
<strong> Input to the model</strong> 
<p>
</p>	
[INST]<<SYS>>
You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]
Prediction: ...
Analysis: ...

<</SYS>>

[Company Introduction]:

Amazon.com Inc is a leading entity in the Retail sector. Incorporated and publicly traded since 1997-05-15, the company has established its reputation as one of the key players in the market. As of today, Amazon.com Inc has a market capitalization of 1585033.68 in USD, with 10334.03 shares outstanding.

Amazon.com Inc operates primarily in the US, trading under the ticker AMZN on the NASDAQ NMS - GLOBAL MARKET. As a dominant force in the Retail space, the company continues to innovate and drive progress within the industry.

From 2023-12-10 to 2023-12-17, AMZN's stock price increased from 147.42 to 149.97. Company news during this period are listed below:

Headline: Retail's Year in Review
Summary: The key word for retail in 2022 was "inventory." This year, it's "shrink."

Headline: 2 Red-Hot Growth Stocks to Buy in 2023 and Beyond
Summary: These high-octane growth stocks are ready to run in December 2023, and they have the AI-boosted stamina to deliver strong returns for years to come.

Headline: Amazon won't have to pay hundreds of millions in back taxes after winning EU case
Summary: LONDON — Amazon won't have to pay about 250 million euros in back taxes after European Union judges ruled in favor of the U.S. e-commerce giant Thursday, dealing a defeat to the 27-nation bloc...

Headline: 3 No-Brainer Warren Buffett Stocks to Buy Right Now
Summary: It's full steam ahead for these great Buffett stocks.

Headline: Top 5 Artificial Intelligence (AI) Stocks You Need To Know For 2024
Summary: Artificial intelligence was salvation for Wall Street this year and it could heat up in 2024. Click here for top 5 AI stocks for 2024.

Basic Financials:

No basic financial reported.

Based on all the information before 2023-12-17, let's first analyze the positive developments and potential concerns for AMZN. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. Then make your prediction of the AMZN stock price movement for next week (2023-12-17 to 2023-12-24). Provide a summary analysis to support your prediction.[/INST]
</div>
<p>
</p>
<div class="third-textbox" id="middleTextbox" contenteditable="true" >
<strong>Finetuned model output </strong>
<p>
</p>
Positive developments: 1. Growth investors' attention on the company following upbeat articles.
2. The growth of AI-boosted revenues.
3. A decline in tax payments after winning the EU case
4. Attraction and consistently strong demand from investors in 2023

Potential concerns:
1. Unsatisfactory customer experience or delivery challenges in recent times.
2. Regulatory concerns and penalties based on Amazon's business structure.
3. Supply chain bottleneck and delivery challenges.

Prediction & analysis:
Prediction: AMZN will likely increase in price by 1.04% in the upcoming week (December 17-24).

Based on a growing clientele and continuous demand in 2023, an optimistic trader outlook should lead to price increase rather than decreasing investors' interest in the company and share price. Despite the recent positive news, ongoing challenges with delivery may create some uncertainty and slow down the expected growth."
</div>
<p>
</p>
<div class="third-textbox" id="rightTextbox" contenteditable="true">
<strong> Base model - Llama-2 output </strong>
<p>
</p>
Positive Developments:

1. Increased demand for e-commerce: The news article "Retail's Year in Review" highlights the continued growth of e-commerce, which bodes well for Amazon's business. As the largest e-commerce platform in the world, Amazon is likely to benefit from this trend.
2. Innovation and AI-boosted growth: The articles "2 Red-Hot Growth Stocks to Buy in 2023 and Beyond" and "Top 5 Artificial Intelligence (AI) Stocks You Need To Know For 2024" suggest that Amazon is investing heavily in AI and automation, which could lead to increased growth and competitiveness.
3. Tax victory: The news article "Amazon won't have to pay hundreds of millions in back taxes after winning EU case" shows that Amazon has been victorious in a tax dispute with the European Union, which could result in significant cost savings for the company.

Potential Concerns:

1. Competition: The e-commerce market is highly competitive, and Amazon faces competition from other big-box retailers as well as smaller, niche players. This could lead to increased competition and pricing pressure.
2. Economic uncertainty: The articles "3 No-Brainer Warren Buffett Stocks to Buy Right Now" and "Top 5 Artificial Intelligence (AI) Stocks You Need To Know For 2024" mention that investors are becoming more cautious in the face of economic uncertainty, which could impact Amazon's stock price.
3. Regulatory challenges: Amazon has faced increased scrutiny from regulators in recent years, particularly in the areas of data privacy, monopolistic practices, and tax compliance. Any adverse regulatory developments could negatively impact the company's stock price.

Based on these factors, my prediction for Amazon's stock price movement for the upcoming week (2023-12-17 to 2023-12-24) is for a slight decrease. Although the positive developments mentioned above suggest that Amazon's business is likely to continue to grow, the potential concerns around competition and economic uncertainty could offset these gains and lead to a slight decline in the stock price.

Analysis:

Amazon's stock price has been steadily increasing over the past few months, driven by the company's innovation and growth in the e-commerce space. However, the recent economic uncertainty and increased competition in the retail space could lead to a slowdown in Amazon's growth momentum. While the company's tax victory and investments in AI and automation could provide long-term benefits, these gains may not be enough to offset the near-term challenges facing the company. Therefore, my prediction for the next week is for a slight decrease in Amazon's stock price.	
</div>


Code for the project can be found on [github](https://github.com/suhaspillai/LLM_Oracle_of_Ohama)








