
基于trainer实现的finetune代码，可以训练huggingface上的生成大模型。

SFT Data Example

{"text": "\n\nHuman: 为一本关于被红眼生物诅咒的家族的恐怖小说生成一个引人入胜且独特的标题。\n\nAssistant: 《鬼怪之眼：被诅咒的血脉》"}
{"text": "\n\nHuman: 用“勇气”、“诱惑”和“命运”这三个词写一首诗。\n\nAssistant: 雄心壮志，奋发有为，\n勇气汇聚心间，\n勇往直前。。。"}

or zju-7B
{"text": "<\s>Human: 为一本关于被红眼生物诅咒的家族的恐怖小说生成一个引人入胜且独特的标题。<\s>Assistant: 《鬼怪之眼：被诅咒的血脉》"}


Pretraining Data Example

{"text": "为一本关于被红眼生物诅咒的家族的恐怖小说生成一个引人入胜且独特的标题。"}
{"text": "这个算法的工作原理是循环遍历数组中的每个元素，找到它后面所有未排序元素中的最小值"}
