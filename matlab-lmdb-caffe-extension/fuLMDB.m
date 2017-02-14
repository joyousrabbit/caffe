classdef fuLMDB < handle
    %FULMDB Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        count = 0;
        database;
        transaction;
    end
    
    methods
        function obj = fuLMDB(dbPath,mapSize)
            obj.database = lmdb.DB(dbPath,'MAPSIZE',mapSize);
            obj.transaction = obj.database.begin();
        end
        
        function insert(obj,value)
            obj.count = obj.count+1;
            obj.transaction.put(sprintf('%08d',obj.count),value);
            if mod(obj.count,1000)==0
                obj.transaction.commit();
                obj.transaction = obj.database.begin();
            end
        end
        
        function commit(obj)
            obj.transaction.commit();
        end
        
        function delete(obj)
            obj.transaction.commit();
            clear obj.transaction;
            clear obj.database;
        end
    end
    
end

