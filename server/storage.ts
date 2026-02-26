export interface IStorage {
  // Empty storage as we don't need persistence for this app
}

export class MemStorage implements IStorage {
  constructor() {}
}

export const storage = new MemStorage();
