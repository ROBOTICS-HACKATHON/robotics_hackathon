export interface SliderChallenge {
  id: string;
  src: string;
  alt: string;
  targetX: number;
  targetY: number;
  pieceSize: number;
  tolerance: number;
}

export const sliderChallenges = [
  { id: 'slider-1', src: '/images/slider/puzzle-1.jpg', alt: 'Mountains reflected in a blue lake', targetX: 0.62, targetY: 0.35, pieceSize: 0.16, tolerance: 0.025 },
  { id: 'slider-2', src: '/images/slider/puzzle-2.jpg', alt: 'Sunlit peaks and pine forest reflected in a turquoise lake', targetX: 0.45, targetY: 0.42, pieceSize: 0.18, tolerance: 0.025 },
  { id: 'slider-3', src: '/images/slider/puzzle-3.jpg', alt: 'Sunbeams over a mountain valley and lake', targetX: 0.71, targetY: 0.26, pieceSize: 0.15, tolerance: 0.025 },
  { id: 'slider-4', src: '/images/slider/puzzle-4.jpg', alt: 'A waterfall beside a green hillside at sunset', targetX: 0.55, targetY: 0.32, pieceSize: 0.17, tolerance: 0.025 },
  { id: 'slider-5', src: '/images/slider/puzzle-5.jpg', alt: 'Snow-covered mountain peaks', targetX: 0.35, targetY: 0.3, pieceSize: 0.19, tolerance: 0.025 },
  { id: 'slider-6', src: '/images/slider/puzzle-6.jpg', alt: 'A misty lake between steep mountains', targetX: 0.64, targetY: 0.45, pieceSize: 0.16, tolerance: 0.025 },
  { id: 'slider-7', src: '/images/slider/puzzle-7.jpg', alt: 'Stars above a snowy mountain range', targetX: 0.47, targetY: 0.4, pieceSize: 0.18, tolerance: 0.025 },
  { id: 'slider-8', src: '/images/slider/puzzle-8.jpg', alt: 'A person reaching toward the camera in black and white', targetX: 0.68, targetY: 0.3, pieceSize: 0.15, tolerance: 0.025 },
  { id: 'slider-9', src: '/images/slider/puzzle-9.jpg', alt: 'Northern lights over snowy mountains', targetX: 0.39, targetY: 0.37, pieceSize: 0.17, tolerance: 0.025 },
  { id: 'slider-10', src: '/images/slider/puzzle-10.jpg', alt: 'A turquoise mountain lake framed by pine branches', targetX: 0.57, targetY: 0.25, pieceSize: 0.2, tolerance: 0.025 },
] as const satisfies readonly SliderChallenge[];
