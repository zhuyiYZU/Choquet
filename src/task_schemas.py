DEFAULT_LABEL_SCHEMA = {
    "class_0": "不属于目标类别 / negative class",
    "class_1": "属于目标类别 / positive class",
}


TASK_LABEL_SCHEMAS = {
    # ======================
    # Binary classification tasks
    # ======================
    "clickbait": {
        "class_0": "非标题党 / 非点击诱导 / 客观普通标题",
        "class_1": "标题党 / 点击诱导 / 具有误导性或诱导点击倾向",
    },

    "fake_news": {
        "class_0": "真实新闻 / 可信信息 / 无明显虚假或误导特征",
        "class_1": "假新闻 / 虚假信息 / 误导性或难以核实的信息",
    },

    "implicit_sentiment": {
        "class_0": "不包含隐式情感 / 无明显隐含态度倾向",
        "class_1": "包含隐式情感 / 存在隐含情绪或态度倾向",
    },

    # ======================
    # Multi-class short text classification tasks
    # ======================

    # TMNnews: 7 classes
    "tmnnews": {
        "class_0": "business / 商业、财经、公司、市场、投资相关短文本",
        "class_1": "entertainment / 娱乐、电影、电视、音乐、明星、文化活动相关短文本",
        "class_2": "health / 健康、医疗、疾病、药物、公共卫生相关短文本",
        "class_3": "sci_tech / 科学技术、互联网、软件、硬件、科研发现相关短文本",
        "class_4": "sport / 体育、比赛、球队、运动员、赛事相关短文本",
        "class_5": "us / 美国国内新闻、美国社会、法律、地方事件相关短文本",
        "class_6": "world / 国际新闻、全球政治、外交、战争、地区冲突相关短文本",
    },

    # Snippets: 8 classes
    "snippets": {
        "class_0": "business / 商业、金融、公司、交易、市场、管理相关短文本",
        "class_1": "computers / 计算机、网络、软件、硬件、互联网技术相关短文本",
        "class_2": "culture-arts-entertainment / 文化、艺术、娱乐、电影、音乐、文学相关短文本",
        "class_3": "education-science / 教育、科学、研究、学校、学术知识相关短文本",
        "class_4": "engineering / 工程、机械、电气、制造、技术系统相关短文本",
        "class_5": "health / 健康、医学、疾病、治疗、公共卫生相关短文本",
        "class_6": "politics-society / 政治、政府、社会问题、公共事务相关短文本",
        "class_7": "sports / 体育、运动、比赛、运动员、赛事相关短文本",
    },
}


# Stable ASCII-only schemas used for cache keys and prompts.
# Keep these values conservative and version-stable: label_schema participates
# in the strict LLM cache primary key.
DEFAULT_LABEL_SCHEMA = {
    "class_0": "negative_or_not_target",
    "class_1": "positive_or_target",
}

TASK_LABEL_SCHEMAS = {
    "clickbait": {
        "class_0": "non_clickbait_objective_headline",
        "class_1": "clickbait_misleading_or_curiosity_gap_headline",
    },
    "fake_news": {
        "class_0": "real_news_credible_information",
        "class_1": "fake_news_false_or_misleading_information",
    },
    "implicit_sentiment": {
        "class_0": "no_implicit_sentiment_or_no_clear_attitude",
        "class_1": "contains_implicit_sentiment_or_hidden_attitude",
    },
    "tmnnews": {
        "class_0": "business_finance_company_market",
        "class_1": "entertainment_movies_tv_music_culture",
        "class_2": "health_medicine_public_health",
        "class_3": "science_technology_internet_research",
        "class_4": "sports_games_teams_athletes",
        "class_5": "us_domestic_news_society_law",
        "class_6": "world_international_politics_conflict",
    },
    "snippets": {
        "class_0": "business_finance_company_management",
        "class_1": "computers_network_software_hardware",
        "class_2": "culture_arts_entertainment",
        "class_3": "education_science_research",
        "class_4": "engineering_manufacturing_systems",
        "class_5": "health_medicine_public_health",
        "class_6": "politics_society_public_affairs",
        "class_7": "sports_games_athletes_events",
    },
}


TASK_ALIASES = {
    # ======================
    # Clickbait
    # ======================
    "clickbait": "clickbait",
    "clickbait_detection": "clickbait",
    "标题党": "clickbait",
    "点击诱导": "clickbait",

    # ======================
    # Fake news
    # ======================
    "fake_news": "fake_news",
    "fake news": "fake_news",
    "fake_news_detection": "fake_news",
    "假新闻": "fake_news",
    "虚假信息": "fake_news",

    # ======================
    # Implicit sentiment
    # ======================
    "implicit_sentiment": "implicit_sentiment",
    "implicit sentiment": "implicit_sentiment",
    "implicit_sentiment_detection": "implicit_sentiment",
    "implicit sentiment detection": "implicit_sentiment",
    "隐式情感": "implicit_sentiment",

    # ======================
    # TMNnews
    # ======================
    "tmnnews": "tmnnews",
    "tmn_news": "tmnnews",
    "tmn news": "tmnnews",
    "tmn": "tmnnews",

    # ======================
    # Snippets
    # ======================
    "snippets": "snippets",
    "snippet": "snippets",
    "web snippets": "snippets",
    "web_snippets": "snippets",

    # ======================
    # General short text aliases
    # 注意：short_text 不直接映射到某一个 schema，
    # 因为 TMNnews 是 7 类，Snippets 是 8 类。
    # 如果所有数据都写 short_text，会无法判断到底该用 7 类还是 8 类。
    # ======================
}


# ASCII aliases avoid future source-encoding drift from changing schema lookup.
TASK_ALIASES = {
    "clickbait": "clickbait",
    "clickbait_detection": "clickbait",
    "fake_news": "fake_news",
    "fake news": "fake_news",
    "fake_news_detection": "fake_news",
    "implicit_sentiment": "implicit_sentiment",
    "implicit sentiment": "implicit_sentiment",
    "implicit_sentiment_detection": "implicit_sentiment",
    "implicit sentiment detection": "implicit_sentiment",
    "tmnnews": "tmnnews",
    "tmn_news": "tmnnews",
    "tmn news": "tmnnews",
    "tmn": "tmnnews",
    "snippets": "snippets",
    "snippet": "snippets",
    "web snippets": "snippets",
    "web_snippets": "snippets",
}


def normalize_task_name(task_name: object) -> str:
    text = str(task_name or "").strip().lower().replace("-", "_")
    return TASK_ALIASES.get(text, text)


def get_task_label_schema(
    task_name: object = None,
    task_description: object = None,
) -> dict:
    normalized = normalize_task_name(task_name)

    if normalized in TASK_LABEL_SCHEMAS:
        return TASK_LABEL_SCHEMAS[normalized]

    description = str(task_description or "").strip().lower()
    description_normalized = description.replace("-", "_")

    for alias, schema_key in TASK_ALIASES.items():
        alias_normalized = alias.lower().replace("-", "_")
        if alias.lower() in description or alias_normalized in description_normalized:
            return TASK_LABEL_SCHEMAS[schema_key]

    return DEFAULT_LABEL_SCHEMA


def get_num_classes(
    task_name: object = None,
    task_description: object = None,
) -> int:
    """
    Return the number of classes for the current task.
    """
    return len(get_task_label_schema(task_name, task_description))


def format_label_schema_for_prompt(
    task_name: object = None,
    task_description: object = None,
) -> str:
    """
    Format label schema into human-readable class declarations for LLM prompt.
    Example:
    class_0: business
    class_1: entertainment
    """
    schema = get_task_label_schema(task_name, task_description)
    lines = []

    for class_key, class_desc in schema.items():
        lines.append(f"{class_key}: {class_desc}")

    return "\n".join(lines)


def build_probability_json_template(
    task_name: object = None,
    task_description: object = None,
) -> dict:
    """
    Build a dynamic probability output template for LLM prompt.
    Binary task:
    {
      "class_0_probability": 0.0,
      "class_1_probability": 1.0,
      ...
    }

    Multi-class task:
    {
      "class_0_probability": 0.0,
      ...
      "class_7_probability": 0.0,
      ...
    }
    """
    schema = get_task_label_schema(task_name, task_description)

    output = {}
    for class_key in schema.keys():
        output[f"{class_key}_probability"] = 0.0

    output["confidence"] = 0.0
    output["explanation"] = "不超过50字的中文解释"

    return output


def get_schema_key(
    task_name: object = None,
    task_description: object = None,
) -> str:
    """Return a stable, cache-friendly representation of the label schema."""
    schema = get_task_label_schema(task_name, task_description)
    return "|".join(f"{key}={value}" for key, value in schema.items())
