
import { Request, Response } from 'express';

// Placeholder for menu items - we'll replace this with Square API data later
const menuItems = [
    { id: '1', name: 'Sunrise Smoothie', description: 'Mango, pineapple, banana', price: 8.50, category: 'Smoothies', imageName: 'smoothie1' },
    { id: '2', name: 'Green Machine', description: 'Kale, spinach, apple, cucumber', price: 9.00, category: 'Juices', imageName: 'juice1' },
    { id: '3', name: 'Acai Bowl', description: 'Acai, granola, banana, berries', price: 12.00, category: 'Bowls', imageName: 'bowl1' },
    { id: '4', name: 'Pitaya Bowl', description: 'Pitaya, granola, coconut, kiwi', price: 12.50, category: 'Bowls', imageName: 'bowl2' },
    { id: '5', name: 'Energy Bites', description: 'Oats, dates, chia seeds', price: 5.00, category: 'Snacks', imageName: 'snack1' },
];

export const getMenu = (req: Request, res: Response) => {
    res.json(menuItems);
};
