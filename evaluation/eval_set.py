import sys
sys.path.insert(0, '/home/smslab1/PycharmProjects/AnomalyDetection/Eagle')
from GPT4.gpt4v import GPT4Query, instruction_P, instruction_sample
from qwen_vl_utils import process_vision_info

class QwenQuery(GPT4Query):
    def __init__(self, image_path, text_gt, processor, model,
                 few_shot=[], visualization=False, agent=None,
                 domain_knowledge=None, args=None,
                 mask_path=None, CoT=None, Prompt=None,
                 defect_shot=None, status=None,
                 red_enhanced_prompt=None, image_info=None,
                 class_name=None,apply_attention=None):
        super(QwenQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.processor = processor
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.mask_path = mask_path
        self.CoT = CoT
        self.args = args
        self.class_name = class_name
        self.Prompt = Prompt
        self.status = status
        self.image_info = image_info
        self.apply_attention = apply_attention
    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            if self.args.record_history:
                pass
            else:
                content = self.get_query(part_questions)
                if self.apply_attention==True:
                    response = self.model.generate_response_attention(content)
                else:
                    response = self.model.generate_response(content)
            print(response)
            gpt_answer = self.parse_answer(response, part_questions[0]['options'])
            if len(gpt_answer) == 0:
                gpt_answer.append(response)
                logging.error(f"No matching answer at {self.image_path}: {part_questions}")
            gpt_answers.append(gpt_answer[-1])

        return questions, answers, gpt_answers

    def get_query(self, conversation):
        incontext = []
        question_type = conversation[0]["type"]
        if self.few_shot:
            incontext.append({
                "type": "text",
                "text": f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried."
            })
        for ref_image_path in self.few_shot:
            ref_image = cv2.imread(ref_image_path)
            if self.visualization:
                self.visualize_image(ref_image)

            incontext.append({"type": "image","image": ref_image_path})

        payload = self.model.processor_framework(instruction_sample, conversation, self.image_path, self.image_info)
        return payload

