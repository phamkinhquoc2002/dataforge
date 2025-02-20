from utils import user_prompt_initialize, prompt_initialize
from tasks import Task
from api_providers import GoogleAIModel
def main():
    task = Task(
        task_name="sft",
        grounded_knowledge="""Lionel Messi, widely regarded as one of the greatest footballers of all time, continues to make headlines both on and off the field. 
        Despite leaving FC Barcelona nearly four years ago, his iconic number '10' jersey remains one of the club's top sellers, especially among tourists. 
        Only the jerseys of rising star Lamine Yamal and seasoned striker Robert Lewandowski surpass Messi's in sales. 

        Currently, Messi is leading Inter Miami into a pivotal season under the guidance of new head coach Javier Mascherano. 
        The team is poised to compete for up to five trophies, including the MLS Cup and the FIFA Club World Cup. This season is particularly significant 
        as the club prepares to transition to a new stadium in Miami next year.""",
        num_of_data=5,
        language="Vietnamese",

    )
    message = prompt_initialize(task)
    llm = GoogleAIModel(api_key="AIzaSyDL7NFfqMkn4sTfkofz58UHr3YOrSJPh88")
    output = llm.chat(message)
    print(output)

if __name__ == "__main__":
    main()