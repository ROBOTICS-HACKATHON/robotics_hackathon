export const imageChallenge = {
  prompt: 'Select all images with cats.',
  instruction: 'Select all images with',
  subject: 'cats',
  tiles: [
    { src: '/images/cat-1.jpg', alt: 'A gray cat', correct: true },
    { src: '/images/dog-1.jpg', alt: 'A shiba puppy', correct: false },
    { src: '/images/bird-1.jpg', alt: 'A parrot', correct: false },
    { src: '/images/bird-2.jpg', alt: 'A kingfisher', correct: false },
    { src: '/images/cat-2.jpg', alt: 'An orange cat', correct: true },
    { src: '/images/dog-2.jpg', alt: 'Golden puppies', correct: false },
    { src: '/images/dog-3.jpg', alt: 'A white dog', correct: false },
    { src: '/images/bird-3.jpg', alt: 'A flying bird', correct: false },
    { src: '/images/cat-3.jpg', alt: 'A tabby cat', correct: true },
  ],
} as const;

export const sliderChallenge = {
  src: '/images/landscape.jpg',
  alt: 'Mountains reflected in a lake',
  targetX: 0.62,
  targetY: 0.35,
  pieceSize: 0.16,
  tolerance: 0.025,
} as const;
