import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
from typing import Tuple, List


class ImageClassifier:
    IMAGENET_LABELS = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich", "brambling",
        "goldfinch", "house finch", "junco", "indigo bunting", "robin",
        "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite",
        "bald eagle", "vulture", "great grey owl", "European fire salamander",
        "common newt", "eft", "spotted salamander", "axolotl", "bullfrog",
        "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
        "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
        "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
        "Gila monster", "green lizard", "African chameleon", "Komodo dragon",
        "African crocodile", "American alligator", "triceratops", "thunder snake",
        "ringneck snake", "hognose snake", "green snake", "king snake",
        "garter snake", "water snake", "vine snake", "night snake",
        "boa constrictor", "rock python", "Indian cobra", "green mamba",
        "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
        "harvestman", "scorpion", "black and gold garden spider", "barn spider",
        "garden spider", "black widow", "tarantula", "wolf spider", "tick",
        "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken",
        "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
        "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
        "black swan", "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
        "flatworm", "nematode", "conch", "snail", "slug",
        "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
        "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish",
        "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
        "flamingo", "little blue heron", "American egret", "bittern", "crane",
        "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
        "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
        "king penguin", "albatross", "grey whale", "killer whale", "dugong",
        "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese",
        "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
        "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
        "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
        "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
        "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
        "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier",
        "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier",
        "wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier", "Airedale",
        "cairn", "Australian terrier", "Dandie Dinmont", "Boston bull", "miniature schnauzer",
        "giant schnauzer", "standard schnauzer", "Scotch terrier", "Tibetan terrier",
        "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier",
        "Lhasa", "flat-coated retriever", "curly-coated retriever", "golden retriever",
        "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer",
        "vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel",
        "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel",
        "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael dog",
        "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
        "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres",
        "Rottweiler", "German shepherd", "Doberman", "miniature pinscher",
        "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller",
        "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff",
        "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute",
        "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug",
        "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian",
        "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan",
        "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless",
        "timber wolf", "white wolf", "red wolf", "coyote", "dingo",
        "dhole", "African hunting dog", "hyena", "red fox", "kit fox",
        "Arctic fox", "grey fox", "tabby", "tiger cat", "Persian cat",
        "Siamese cat", "Egyptian cat", "cougar", "lynx", "leopard",
        "snow leopard", "jaguar", "lion", "tiger", "cheetah",
        "brown bear", "American black bear", "ice bear", "sloth bear", "mongoose",
        "meerkat", "tiger beetle", "ladybug", "ground beetle", "long-horned beetle",
        "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly",
        "bee", "ant", "grasshopper", "cricket", "walking stick",
        "cockroach", "mantis", "cicada", "leafhopper", "lacewing",
        "dragonfly", "damselfly", "admiral", "ringlet", "monarch",
        "cabbage butterfly", "sulphur butterfly", "lycaenid", "starfish",
        "sea urchin", "sea cucumber", "cottontail", "hare", "Angora",
        "hamster", "porcupine", "fox squirrel", "marmot", "beaver",
        "guinea pig", "common sorrel", "zebra", "hog", "wild boar",
        "warthog", "hippopotamus", "ox", "water buffalo", "bison",
        "ram", "bighorn", "ibex", "hartebeest", "impala",
        "gazelle", "Arabian camel", "llama", "weasel", "mink",
        "polecat", "black-footed ferret", "otter", "skunk", "badger",
        "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee",
        "gibbon", "siamang", "guenon", "patas", "baboon",
        "macaque", "langur", "colobus", "proboscis monkey", "marmoset",
        "white-headed capuchin", "howler monkey", "titi", "spider monkey",
        "squirrel monkey", "Madagascar cat", "indri", "Indian elephant",
        "African elephant", "lesser panda", "giant panda", "barracouta", "eel",
        "coho", "rock beauty", "anemone fish", "sturgeon", "gar",
        "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship",
        "altar", "ambulance", "amphibian", "analog clock", "apiary",
        "apron", "ashcan", "assault rifle", "backpack", "bakery",
        "balance beam", "balloon", "ballpoint", "Band Aid", "banjo",
        "bannister", "barbell", "barber chair", "barbershop", "barn",
        "barometer", "barrel", "barrow", "baseball", "basketball",
        "bassinet", "bassoon", "bathing cap", "bath towel", "bathtub",
        "beach wagon", "beacon", "beaker", "bearskin", "beer bottle",
        "beer glass", "bell cote", "bib", "bicycle-built-for-two", "bikini",
        "binder", "binoculars", "birdhouse", "boathouse", "bobsled",
        "bolo tie", "bonnet", "bookcase", "bookshop", "bottlecap",
        "bow", "bow tie", "brass", "brassiere", "breakwater",
        "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
        "bullet train", "butcher shop", "cab", "caldron", "candle",
        "cannon", "canoe", "can opener", "cardigan", "car mirror",
        "carousel", "carpenter kit", "carton", "car wheel", "cash machine",
        "cassette", "cassette player", "castle", "catamaran", "CD player",
        "cello", "cellular telephone", "chain", "chainlink fence", "chain mail",
        "chain saw", "chest", "chiffonier", "chime", "china cabinet",
        "Christmas stocking", "church", "cinema", "cleaver", "cliff dwelling",
        "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot",
        "coil", "combination lock", "computer keyboard", "confectionery",
        "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
        "cowboy hat", "cradle", "crane", "crash helmet", "crate",
        "crib", "Crock Pot", "croquet ball", "crutch", "cuirass",
        "dam", "desk", "desktop computer", "dial telephone", "diaper",
        "digital clock", "digital watch", "dining table", "dishrag", "dishwasher",
        "disk brake", "dock", "dogsled", "dome", "doormat",
        "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven",
        "electric fan", "electric guitar", "electric locomotive", "entertainment center",
        "envelope", "espresso maker", "face powder", "feather boa", "file",
        "fireboat", "fire engine", "fire screen", "flagpole", "flute",
        "folding chair", "football helmet", "forklift", "fountain", "fountain pen",
        "four-poster", "freight car", "French horn", "frying pan", "fur coat",
        "garbage truck", "gasmask", "gas pump", "goblet", "go-kart",
        "golf ball", "golfcart", "gondola", "gong", "gown",
        "grand piano", "greenhouse", "grille", "grocery store", "guillotine",
        "hair slide", "hair spray", "half track", "hammer", "hamper",
        "hand blower", "hand-held computer", "handkerchief", "hard disc", "harmonica",
        "harp", "harvester", "hatchet", "holster", "home theater",
        "honeycomb", "hook", "hoop skirt", "horizontal bar", "horse cart",
        "hourglass", "iPod", "iron", "jack-o-lantern", "jean",
        "jeep", "jersey", "jigsaw puzzle", "jinrikisha", "joystick",
        "kimono", "knee pad", "knot", "lab coat", "ladle",
        "lampshade", "laptop", "lawn mower", "lens cap", "letter opener",
        "library", "lifeboat", "lighter", "limousine", "liner",
        "lipstick", "Loafer", "lotion", "loudspeaker", "loupe",
        "luggage", "manhole cover", "maraca", "marimba", "mask",
        "matchstick", "maypole", "maze", "measuring cup", "medicine chest",
        "megalith", "microphone", "microwave", "military uniform", "milk can",
        "minibus", "miniskirt", "minivan", "missile", "mitten",
        "mixing bowl", "mobile home", "Model T", "modem", "monastery",
        "monitor", "moped", "mortar", "mortarboard", "mosque",
        "mosquito net", "motor scooter", "mountain bike", "mountain tent", "mouse",
        "mousetrap", "moving van", "muzzle", "nail", "neck brace",
        "necklace", "nipple", "notebook", "obelisk", "oboe",
        "ocarina", "odometer", "oil filter", "organ", "oscilloscope",
        "overskirt", "oxcart", "oxygen mask", "packet", "paddle",
        "paddle wheel", "padlock", "paintbrush", "pajama", "palace",
        "panpipe", "paper towel", "parachute", "parallel bars", "park bench",
        "parking meter", "passenger car", "patio", "pay-phone", "pedestal",
        "pencil box", "pencil sharpener", "perfume", "Petri dish", "photocopier",
        "pick", "pickelhaube", "picket fence", "pickup", "pier",
        "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel",
        "pirate", "pitcher", "plane", "planetarium", "plastic bag",
        "plate rack", "plow", "plunger", "Polaroid camera", "pole",
        "police van", "poncho", "pool table", "pop bottle", "pot",
        "potter wheel", "power drill", "prayer rug", "printer", "prison",
        "projectile", "projector", "puck", "punching bag", "purse",
        "quill", "quilt", "racer", "racket", "radiator",
        "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel",
        "reflex camera", "refrigerator", "remote control", "restaurant", "revolver",
        "rifle", "rocking chair", "rotisserie", "rubber eraser", "rugby ball",
        "rule", "running shoe", "safe", "safety pin", "saltshaker",
        "sandal", "sarong", "sax", "scabbard", "scale",
        "school bus", "schooner", "scoreboard", "screen", "screw",
        "screwdriver", "seat belt", "sewing machine", "shield", "shoe shop",
        "shoji", "shopping basket", "shopping cart", "shovel", "shower cap",
        "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule",
        "sliding door", "slot", "snorkel", "snowmobile", "snowplow",
        "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero",
        "soup bowl", "space bar", "space heater", "space shuttle", "spatula",
        "speedboat", "spider web", "spindle", "sports car", "spotlight",
        "stage", "steam locomotive", "steel arch bridge", "steel drum",
        "stethoscope", "stole", "stone wall", "stopwatch", "stove",
        "strainer", "streetcar", "stretcher", "studio couch", "stupa",
        "submarine", "suit", "sundial", "sunglass", "sunglasses",
        "sunscreen", "suspension bridge", "swab", "sweatshirt", "swimming trunks",
        "swing", "switch", "syringe", "table lamp", "tank",
        "tape player", "teapot", "teddy", "television", "tennis ball",
        "thatch", "theater curtain", "thimble", "thresher", "thumb",
        "thunder", "toilet seat", "torch", "totem pole", "tow truck",
        "toyshop", "tractor", "trailer truck", "tray", "trench coat",
        "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus",
        "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella",
        "unicycle", "upright", "vacuum", "vase", "vault",
        "velvet", "vending machine", "vestment", "viaduct", "violin",
        "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe",
        "warplane", "washbasin", "washer", "water bottle", "water jug",
        "water tower", "whiskey jug", "whistle", "wig", "window screen",
        "window shade", "Windsor tie", "wine bottle", "wing", "wok",
        "wooden spoon", "wool", "worm fence", "wreck", "yawl",
        "yurt", "web site", "comic book", "crossword puzzle", "street sign",
        "traffic light", "book jacket", "menu", "plate", "guacamole",
        "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
        "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog",
        "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
        "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
        "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
        "orange", "lemon", "fig", "pineapple", "banana",
        "jackfruit", "custard apple", "pomegranate", "hay", "carbonara",
        "chocolate syrup", "dough", "meatloaf", "pizza", "potpie",
        "burrito", "red wine", "espresso", "cup", "eggnog",
        "alp", "bubble", "cliff", "coral reef", "geyser",
        "lakeside", "promontory", "sandbar", "seashore", "valley",
        "volcano", "ballplayer", "groom", "scuba diver", "rapeseed",
        "daisy", "yellow lady slipper", "corn", "acorn", "hip",
        "buckeye", "coral fungus", "agaric", "bolete", "stinkhorn",
        "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue"
    ]

    SCENE_CATEGORIES = {
        "outdoor_nature": ["alp", "valley", "volcano", "lakeside", "seashore", "promontory",
                           "cliff", "coral reef", "geyser", "sandbar", "bubble"],
        "city_urban": ["streetcar", "church", "cinema", "bakery", "bookshop", "grocery store",
                       "barbershop", "butcher shop", "restaurant", "toyshop", "library",
                       "monastery", "mosque", "castle", "boathouse", "greenhouse"],
        "indoor_room": ["home theater", "library", "dining table", "desktop computer",
                        "entertainment center", "china cabinet", "bookcase", "four-poster",
                        "cradle", "Crock Pot", "microwave", "refrigerator", "stove",
                        "washing machine", "dishwasher"],
        "road_traffic": ["car wheel", "traffic light", "street sign", "taxi", "ambulance",
                         "fire engine", "garbage truck", "police van", "minibus", "minivan",
                         "passenger car", "recreational vehicle", "moving van", "trailer truck",
                         "tractor", "school bus", "freight car", "electric locomotive",
                         "steam locomotive", "bullet train", "highway", "race car"],
        "water_activity": ["beach wagon", "canoe", "catamaran", "container ship", "dock",
                           "fireboat", "gondola", "liner", "pirate", "submarine",
                           "trimaran", "yawl", "wreck", "speedboat", "motorboat"],
        "air_transport": ["aircraft carrier", "airliner", "airship", "warplane",
                          "space shuttle", "balloon", "parachute"],
        "sports": ["basketball", "baseball", "volleyball", "tennis ball", "soccer ball",
                   "ping-pong ball", "rugby ball", "golf ball", "crash helmet",
                   "football helmet", "balance beam", "parallel bars", "horizontal bar",
                   "punching bag", "ski", "snowmobile", "bobsled", "mountain bike",
                   "racer", "go-kart", "moped", "mountain tent"],
        "food_kitchen": ["pizza", "hotdog", "cheeseburger", "burrito", "potpie", "meatloaf",
                         "espresso", "cup", "wine bottle", "beer bottle", "beer glass",
                         "coffee mug", "red wine", "eggnog", "consomme", "hot pot",
                         "ice cream", "trifle", "carbonara", "chocolate syrup", "dough",
                         "frying pan", "Crock Pot", "espresso maker", "microwave"]
    }

    def __init__(self, model_path: str, use_gpu: bool = False):
        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (input_shape[2], input_shape[3]) if len(input_shape) == 4 else (224, 224)
            self.model_loaded = True
        except Exception as e:
            print(f"[WARN] Failed to load classification model: {e}. Using mock classifier.")
            self.model_loaded = False
            self.input_size = (224, 224)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _map_to_scene(self, label: str) -> str:
        for scene, keywords in self.SCENE_CATEGORIES.items():
            if label in keywords:
                return scene
        return "general"

    def predict(self, frame: np.ndarray, top_k: int = 3) -> Tuple[str, List[dict]]:
        if not self.model_loaded:
            mock_scenes = ["city_urban", "outdoor_nature", "indoor_room", "road_traffic", "water_activity"]
            scene = mock_scenes[hash(str(frame.shape)) % len(mock_scenes)]
            return scene, [{"label": scene, "confidence": 0.85}]

        input_data = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        predictions = outputs[0][0]
        top_indices = predictions.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            label_idx = idx % len(self.IMAGENET_LABELS)
            results.append({
                "label": self.IMAGENET_LABELS[label_idx],
                "confidence": float(predictions[idx])
            })

        scene = self._map_to_scene(results[0]["label"])
        return scene, results
