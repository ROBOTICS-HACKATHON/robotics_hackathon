import { imageChallenges } from './image-challenges';
import { sliderChallenges } from './slider-challenges';

export type TrainingChallenge = {
  id: string;
  kind: 'image' | 'slider';
  index: number;
  split: 'train' | 'test';
};

export const trainingCatalog: readonly TrainingChallenge[] = [
  ...imageChallenges.map((challenge, index) => ({
    id: challenge.id, kind: 'image' as const, index, split: challenge.split,
  })),
  ...sliderChallenges.map((challenge, index) => ({
    id: challenge.id, kind: 'slider' as const, index, split: challenge.split,
  })),
];

export function partitionTrainingCatalog(trainingPercent: number): readonly TrainingChallenge[] {
  const percent = Math.max(10, Math.min(90, Math.round(trainingPercent)));
  const imageTrainCount = Math.round(imageChallenges.length * percent / 100);
  const sliderTrainCount = Math.round(sliderChallenges.length * percent / 100);
  return trainingCatalog.map((challenge) => ({
    ...challenge,
    split: challenge.index < (challenge.kind === 'image' ? imageTrainCount : sliderTrainCount)
      ? 'train' : 'test',
  }));
}
