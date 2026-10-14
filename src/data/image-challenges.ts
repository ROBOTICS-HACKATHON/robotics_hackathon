export type ImageChallenge = {
  id: string;
  instruction: string;
  subject: string;
  tiles: readonly { src: string; alt: string; correct: boolean }[];
};

type Subject = 'cats' | 'dogs' | 'birds' | 'cars' | 'bicycles' | 'flowers';

const photos = {
  cat1: { src: '/images/cat-1.jpg', alt: 'A gray cat', subject: 'cats' },
  cat2: { src: '/images/cat-2.jpg', alt: 'An orange cat', subject: 'cats' },
  cat3: { src: '/images/cat-3.jpg', alt: 'A tabby cat', subject: 'cats' },
  dog1: { src: '/images/dog-1.jpg', alt: 'A shiba puppy', subject: 'dogs' },
  dog2: { src: '/images/dog-2.jpg', alt: 'Golden puppies', subject: 'dogs' },
  dog3: { src: '/images/dog-3.jpg', alt: 'A white dog', subject: 'dogs' },
  bird1: { src: '/images/bird-1.jpg', alt: 'A parrot', subject: 'birds' },
  bird2: { src: '/images/bird-2.jpg', alt: 'A kingfisher', subject: 'birds' },
  bird3: { src: '/images/bird-3.jpg', alt: 'A flying bird', subject: 'birds' },
  car1: { src: '/images/grid/car-1.jpg', alt: 'A gray car in a parking lot', subject: 'cars' },
  car2: { src: '/images/grid/car-2.jpg', alt: 'A red car beside a park', subject: 'cars' },
  car3: { src: '/images/grid/car-3.jpg', alt: 'Sports cars parked on a street', subject: 'cars' },
  bicycle1: { src: '/images/grid/bicycle-1.jpg', alt: 'An orange bicycle beside a wooden door', subject: 'bicycles' },
  bicycle2: { src: '/images/grid/bicycle-2.jpg', alt: 'A red bicycle beside a pole', subject: 'bicycles' },
  bicycle3: { src: '/images/grid/bicycle-3.jpg', alt: 'A row of bicycles beside a street', subject: 'bicycles' },
  flower1: { src: '/images/grid/flower-1.jpg', alt: 'A sunflower and roses', subject: 'flowers' },
  flower2: { src: '/images/grid/flower-2.jpg', alt: 'Orange tulips and pale roses', subject: 'flowers' },
  flower3: { src: '/images/grid/flower-3.jpg', alt: 'Colorful roses and tulips', subject: 'flowers' },
} as const;

type NinePhotos = readonly [
  keyof typeof photos, keyof typeof photos, keyof typeof photos,
  keyof typeof photos, keyof typeof photos, keyof typeof photos,
  keyof typeof photos, keyof typeof photos, keyof typeof photos,
];

function challenge(id: string, subject: Subject, keys: NinePhotos): ImageChallenge {
  return {
    id,
    instruction: 'Select all images with',
    subject,
    tiles: keys.map((key) => {
      const photo = photos[key];
      return { src: photo.src, alt: photo.alt, correct: photo.subject === subject };
    }),
  };
}

export const imageChallenges = [
  challenge('cats-1', 'cats', ['cat1', 'dog1', 'bird1', 'bird2', 'cat2', 'dog2', 'dog3', 'bird3', 'cat3']),
  challenge('dogs-1', 'dogs', ['flower1', 'dog3', 'car2', 'dog1', 'cat2', 'bicycle1', 'bird2', 'dog2', 'flower3']),
  challenge('birds-1', 'birds', ['car1', 'flower2', 'bird2', 'cat3', 'bicycle2', 'dog1', 'bird3', 'flower1', 'bird1']),
  challenge('cars-1', 'cars', ['car2', 'dog2', 'flower3', 'cat1', 'bird3', 'car3', 'bicycle1', 'car1', 'flower2']),
  challenge('bicycles-1', 'bicycles', ['cat2', 'bicycle3', 'flower1', 'bicycle1', 'car1', 'bird1', 'dog3', 'flower2', 'bicycle2']),
  challenge('flowers-1', 'flowers', ['bicycle2', 'car3', 'flower2', 'bird2', 'flower3', 'cat3', 'flower1', 'dog1', 'bicycle1']),
  challenge('cats-2', 'cats', ['dog2', 'flower2', 'cat3', 'car1', 'bird1', 'bicycle2', 'cat1', 'dog3', 'flower3']),
  challenge('dogs-2', 'dogs', ['dog2', 'car2', 'dog1', 'flower1', 'bird3', 'cat2', 'bicycle3', 'flower3', 'dog3']),
  challenge('cars-2', 'cars', ['bird1', 'car3', 'bicycle2', 'car1', 'flower2', 'cat3', 'car2', 'dog2', 'dog1']),
  challenge('flowers-2', 'flowers', ['flower3', 'cat1', 'bird3', 'bicycle1', 'dog3', 'flower1', 'car2', 'bird2', 'cat2']),
] as const;
